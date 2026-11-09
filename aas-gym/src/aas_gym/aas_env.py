import numpy as np
import gymnasium as gym
import docker
import zmq
import time
import struct
import os
import subprocess
import shutil
import concurrent.futures

from docker.types import DeviceRequest


class AASEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    # Reward/termination tuning (see _update_from_payload)
    GOAL_RADIUS_M = 3.0       # Horizontal radius to consider the goal reached
    GOAL_ALT_TOL_M = 4.0      # Vertical tolerance on top of it
    CRASH_AGL_M = 1.0         # World altitude below which the episode ends (ground contact)
    SUCCESS_REWARD = 100.0
    CRASH_REWARD = -100.0     # Applied when hitting the ground fast, see _update_from_payload
    SOFT_LAND_REWARD = -30.0  # Applied when ending on the ground slowly
    OUT_OF_BOUNDS_REWARD = -50.0
    ALIVE_PENALTY_PER_SEC = -2.5
    PROGRESS_GAIN = 1.0       # Potential shaping gain (meters of progress toward goal)

    def __init__(self,
            instance: int=0,
            gym_freq_hz: int=50,
            autopilot: str="px4",
            camera: bool=True,
            lidar: bool=True,
            odom: str="none",
            num_quads: int=1,
            render_mode=None,
            gpu: bool=True,
            container_engine: str="docker",  # "docker" or "podman" ('pip install podman' + a running podman API socket)
            gpu_vendor: str="auto",          # GPU passthrough strategy: "auto" (detect; none if no GPU found), "nvidia", "amd" or "none"
            max_speed: float=7.0,      # [m/s] commanded by an action of 1.0 (planar and vertical)
            max_yaw_rate: float=1.0,   # [rad/s]
            goal_range=(15.0, 45.0),   # [m] min/max distance of the sampled goal from the spawn point
            bounds_radius_m: float=250.0  # Horizontal distance from spawn beyond which the episode ends
        ):
        super().__init__()

        self.GYM_FREQ_HZ = gym_freq_hz
        self.GYM_INIT_DURATION = 80.0  # Seconds to run unpaused during reset (seconds)
        self.MAX_EPISODE_LENGTH_SEC = 300.0  # Max episode length in seconds (excluding init duration)

        self.NUM_DRONES = int(num_quads)  # Keep in sync with the (currently zero) NUM_VTOLS/NUM_TAILS below
        self.max_speed = float(max_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.goal_range = goal_range
        self.bounds_radius_m = float(bounds_radius_m)

        # Action Space: normalized [vE, vN, vUp (m/s), yaw rate (rad/s)] command for the first quad,
        # scaled by max_speed / max_yaw_rate and flown by MAVROS in GUIDED mode (AUTOPILOT=ardupilot only)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,), dtype=np.float32
        )
        # Observation Space, all float32 in the Gazebo world frame (ENU, meters):
        #   [own position x,y,z; own velocity vx,vy,vz (estimated from ground-truth pose deltas);
        #    goal offset dx,dy,dz] and then, for each additional drone, its position relative to own
        self.obs_dim = 9 + 3 * max(self.NUM_DRONES - 1, 0)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        # Initialize storage for the clock and RL state
        self.sim_sec = 0.0
        self.sim_nanosec = 0.0
        self.gz_model_names = self._gz_model_names(autopilot, num_quads)
        if len(self.gz_model_names) == 0:
            raise ValueError("AASEnv requires at least one drone")
        self.reply_fmt = '<iI' + 'd' * (7 * len(self.gz_model_names))
        self.pos_all = np.zeros((len(self.gz_model_names), 3), dtype=np.float64)  # All drones' world positions
        self.prev_pos = None       # Own previous pose, for velocity estimation
        self.prev_sim_time = None  # Clock [s] at the previous observation
        self.vel_est = np.zeros(3, dtype=np.float64)
        self.spawn_xy = np.zeros(2, dtype=np.float64)
        self.goal = np.zeros(3, dtype=np.float64)
        self.last_dt = 1.0 / self.GYM_FREQ_HZ

        self.max_steps = int(self.MAX_EPISODE_LENGTH_SEC*self.GYM_FREQ_HZ)  # Max steps per episode
        self.step_count = 0
        self.last_event = None  # How the last episode ended: success, crash, soft_land, out_of_bounds or timeout
        
        # Rendering
        self.render_mode = render_mode

        # AAS Setup
        self.HEADLESS = False if self.render_mode == "human" else True # Only display GUIs if render_mode is "human"
        self.AUTOPILOT = autopilot
        if autopilot == "betaflight": # Betaflight SITL clock keeps running while Gazebo is paused between steps, and RTF is locked to 1
            raise ValueError("autopilot='betaflight' is not supported: the Betaflight SITL is not lockstepped with Gazebo")
        if autopilot != "ardupilot":
            print(f"Warning: RL velocity actuation (MAVROS GUIDED mode) is only wired for autopilot='ardupilot'; "
                  f"with autopilot='{autopilot}' actions will be sent to the sim bridge but will not move the drone.")
        self.CAMERA = camera
        self.LIDAR = lidar
        self.ODOM = odom
        self.NUM_QUADS = num_quads
        self.NUM_VTOLS = 0 # TODO add support beyond multicopters
        self.NUM_TAILS = 0 # TODO add support beyond multicopters
        self.WORLD = "impalpable_greyness"
        #
        self.SIM_SUBNET = "10.42"
        self.AIR_SUBNET = "10.22"
        self.SIM_ID = "100"
        # self.GROUND_ID = "101" # Unused
        #
        self.GND_CONTAINER = False # Do NOT use the ground-image to run Zenoh (nor QGC)
        self.RTF = 0.0 # As fast as possible (capped to 15x in the simulation.yml.erb)
        self.START_AS_PAUSED = True # Start the simulation paused and manually step with gz-sim WorldControl
        self.INSTANCE = instance
        #
        sim_parts = self.SIM_SUBNET.split('.')
        self.SIM_SUBNET = f"{sim_parts[0]}.{int(sim_parts[1]) + self.INSTANCE}"
        # air_parts = self.AIR_SUBNET.split('.') # Unused
        # self.AIR_SUBNET = f"{air_parts[0]}.{int(air_parts[1]) + self.INSTANCE}" # Unused
        #
        self.SIM_NET_NAME = f"aas-sim-network-inst{self.INSTANCE}"
        # self.AIR_NET_NAME = f"aas-air-network-inst{self.INSTANCE}" # Unused
        self.SIM_CONT_NAME = f"simulation-container-inst{self.INSTANCE}"
        # self.GND_CONT_NAME = f"ground-container-inst{self.INSTANCE}" # Unused

        # X Server access (this is redundancy for configure_host_x11() in gym_run.py)
        if shutil.which("xhost"):
            try:
                current_acls = subprocess.check_output(["xhost"], text=True)
                if "LOCAL:" not in current_acls and "local:docker" not in current_acls:
                    print("Granting X Server access to Docker containers...")
                    subprocess.run(["xhost", "+local:docker"], check=True)
                    print("X Server access granted.")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not configure xhost: {e}")

        # Container engine setup (Docker or Podman). Podman is driven through the podman-py client, which mirrors
        # the Docker SDK API but raises its own exception classes (not subclasses of docker.errors.NotFound/APIError)
        self.CONTAINER_ENGINE = str(container_engine).strip().lower()
        if self.CONTAINER_ENGINE not in ("docker", "podman"):
            raise ValueError(f"Invalid container_engine '{container_engine}': choose 'docker' or 'podman'")
        self.GPU_VENDOR = str(gpu_vendor).strip().lower()
        if self.GPU_VENDOR not in ("auto", "nvidia", "amd", "none"):
            raise ValueError(f"Invalid gpu_vendor '{gpu_vendor}': choose 'auto', 'nvidia', 'amd' or 'none'")
        try:
            if self.CONTAINER_ENGINE == "podman":
                print("Using container engine: Podman")
                self.client = self._create_podman_client()
                import podman.errors as engine_errors
            else:
                print("Using container engine: Docker")
                try:
                    self.client = docker.from_env()
                except Exception as e:
                    raise RuntimeError("Could not connect to the Docker daemon. Ensure Docker is running.") from e
                engine_errors = docker.errors
        except ImportError as e: # podman pip package missing, but only if Podman was requested
            raise RuntimeError("container_engine='podman' requires the Podman Python client: pip install podman") from e
        self._not_found_exc = (engine_errors.NotFound,) # Engine-appropriate exception classes for the cleanup paths below
        self._api_error_exc = (engine_errors.APIError,)
        #
        def force_container_cleanup(name): # Needed to use AsyncVectorEnv
            try:
                old_container = self.client.containers.get(name)
                print(f"Found existing container '{name}'. Removing it...")
                old_container.remove(force=True)
                time.sleep(1.0)
            except self._not_found_exc:
                pass
            except Exception as e:
                print(f"Warning during cleanup of {name}: {e}")
        #
        networks_config = [
            {"name": self.SIM_NET_NAME, "subnet_base": self.SIM_SUBNET},
            # {"name": self.AIR_NET_NAME, "subnet_base": self.AIR_SUBNET} # Unused
        ]
        self.networks = {}
        for net_config in networks_config:
            net_name = net_config["name"]
            base_ip = net_config["subnet_base"]
            print(f"Setting up Docker Network: {net_name}...")
            try:
                existing_network = self.client.networks.get(net_name)
                existing_network.remove()
                print(f"Existing network '{net_name}' removed.")
            except self._not_found_exc:
                pass
            except self._api_error_exc as e:
                print(f"Warning: Could not remove {net_name}: {e}")
            ipam_pool = docker.types.IPAMPool(
                subnet=f"{base_ip}.0.0/16",
                gateway=f"{base_ip}.0.1"
            )
            ipam_config = docker.types.IPAMConfig(
                pool_configs=[ipam_pool]
            )
            new_network = self.client.networks.create(
                net_name,
                driver="bridge",
                ipam=ipam_config
            )
            self.networks[net_name] = new_network
            print(f"Network '{net_name}' created on subnet {base_ip}.0.0/16")
        #
        env_display = os.environ.get('DISPLAY', '')
        env_xdg = os.environ.get('XDG_RUNTIME_DIR', '')
        gpu_requests, device_binds, gpu_env = self._gpu_container_config(gpu)
        # On SELinux hosts (common with Podman), bind-mounting /tmp/.X11-unix and the ZMQ IPC dir into containers
        # requires disabling label enforcement for these privileged containers
        extra_container_kwargs = {}
        if self.CONTAINER_ENGINE == "podman" and os.path.exists("/sys/fs/selinux"):
            extra_container_kwargs["security_opt"] = ["label=disable"]
        self.ZMQ_IPC_SOCKET_DIR = '/tmp/aas_zmq_sockets'
        os.makedirs(self.ZMQ_IPC_SOCKET_DIR, exist_ok=True)
        volume_binds = {
            '/tmp/.X11-unix': {'bind': '/tmp/.X11-unix', 'mode': 'rw'}, # Replaces "--volume /tmp/.X11-unix:/tmp/.X11-unix:rw"
            self.ZMQ_IPC_SOCKET_DIR: {'bind': self.ZMQ_IPC_SOCKET_DIR, 'mode': 'rw'} # For ZMQ IPC sockets
        }
        #
        force_container_cleanup(self.SIM_CONT_NAME)
        print(f"Creating Simulation Container ({self.SIM_CONT_NAME})...")
        self.simulation_container = self.client.containers.create(
            "simulation-image:latest",
            name=self.SIM_CONT_NAME,
            tty=True, # Replaces -it
            detach=True,
            auto_remove=False,
            privileged=True, # Replaces --privileged
            volumes=volume_binds,
            devices=device_binds,
            device_requests=gpu_requests,
            environment={
                "DISPLAY": env_display,
                "QT_X11_NO_MITSHM": "1",
                **gpu_env,
                "XDG_RUNTIME_DIR": env_xdg,
                "GST_DEBUG": "3",
                "AUTOPILOT": self.AUTOPILOT,
                "HEADLESS": str(self.HEADLESS).lower(),
                "CAMERA": str(self.CAMERA).lower(),
                "LIDAR": str(self.LIDAR).lower(),
                "NUM_QUADS": str(self.NUM_QUADS),
                "NUM_VTOLS": str(self.NUM_VTOLS),
                "NUM_TAILS": str(self.NUM_TAILS),
                "WORLD": self.WORLD,
                "SIMULATED_TIME": "true",
                "RTF": str(self.RTF),
                "START_AS_PAUSED": str(self.START_AS_PAUSED).lower(),
                "SIM_SUBNET": self.SIM_SUBNET,
                # "GROUND_ID": self.GROUND_ID,
                "GND_CONTAINER": str(self.GND_CONTAINER).lower(),
                "ROS_DOMAIN_ID": self.SIM_ID,
                # "HOST_INPUT_GID": "", # Only necessary to let qgcuser have permissions over /dev/input/, e.g., to use a joystick
                "GYMNASIUM" : "true",
                "GYM_FREQ_HZ" : str(self.GYM_FREQ_HZ),
                "GYM_INIT_DURATION" : str(self.GYM_INIT_DURATION),
                "INSTANCE": str(self.INSTANCE),
            },
            **extra_container_kwargs
        )
        print(f"Connecting {self.SIM_CONT_NAME} to {self.SIM_NET_NAME}...")
        self.networks[self.SIM_NET_NAME].connect(
            self.simulation_container,
            ipv4_address=f"{self.SIM_SUBNET}.90.{self.SIM_ID}"
        )
        # print(f"Connecting {self.SIM_CONT_NAME} to {self.AIR_NET_NAME}...")
        # self.networks[self.AIR_NET_NAME].connect(
        #     self.simulation_container,
        #     ipv4_address=f"{self.AIR_SUBNET}.90.{self.SIM_ID}"
        # )
        # self.simulation_container.start()
        #
        self.drone_types = (
            ["quad"] * self.NUM_QUADS +
            ["vtol"] * self.NUM_VTOLS +
            ["tail"] * self.NUM_TAILS
        )
        self.aircraft_containers = []
        for drone_id, drone_type in enumerate(self.drone_types, start=1):
            air_cont_name = f"aircraft-container-inst{self.INSTANCE}_{drone_id}"
            force_container_cleanup(air_cont_name)
            print(f"Creating Aircraft Container {air_cont_name}...")
            air_cont = self.client.containers.create(
                "aircraft-image:latest",
                name=air_cont_name,
                tty=True, # Replaces -it
                detach=True,
                auto_remove=False,
                privileged=True, # Replaces --privileged
                volumes=volume_binds,
                devices=device_binds,
                device_requests=gpu_requests,
                environment={
                    "DISPLAY": env_display,
                    "QT_X11_NO_MITSHM": "1",
                    **gpu_env,
                    "XDG_RUNTIME_DIR": env_xdg,
                    "GST_DEBUG": "3",
                    "AUTOPILOT": self.AUTOPILOT,
                    "HEADLESS": str(self.HEADLESS).lower(),
                    "CAMERA": str(self.CAMERA).lower(),
                    "LIDAR": str(self.LIDAR).lower(),
                    "ODOM": self.ODOM,
                    "DRONE_TYPE": drone_type,
                    "DRONE_ID": str(drone_id),
                    "SIMULATED_TIME": "true",
                    "SIM_SUBNET": self.SIM_SUBNET,
                    # "AIR_SUBNET": self.AIR_SUBNET,
                    "SIM_ID": self.SIM_ID,
                    # "GROUND_ID": self.GROUND_ID,
                    "GND_CONTAINER": str(self.GND_CONTAINER).lower(),
                    "ROS_DOMAIN_ID": str(drone_id),
                    "GYMNASIUM" : "true",
                },
                **extra_container_kwargs
            )
            print(f"Connecting {air_cont_name} to {self.SIM_NET_NAME}...")
            self.networks[self.SIM_NET_NAME].connect(
                air_cont,
                ipv4_address=f"{self.SIM_SUBNET}.90.{drone_id}"
            )
            # print(f"Connecting {air_cont_name} to {self.AIR_NET_NAME}...")
            # self.networks[self.AIR_NET_NAME].connect(
            #     air_cont,
            #     ipv4_address=f"{self.AIR_SUBNET}.90.{drone_id}"
            # )
            # air_cont.start()
            self.aircraft_containers.append(air_cont)
        print("Docker setup complete. All containers are running and connected.")

        # ZeroMQ setup
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.ZMQ_TRANSPORT = "ipc" # "tcp" or "ipc", also uncomment the corresponding block in zeromq_bridge.cpp
        if self.ZMQ_TRANSPORT == "tcp":
            self.ZMQ_PORT = 5555
            self.ZMQ_IP = f"{self.SIM_SUBNET}.90.{self.SIM_ID}"

    @staticmethod
    def _create_podman_client():
        # The Podman API socket can be forced with the PODMAN_URI env var (e.g. "http+unix:///run/user/1000/podman/podman.sock"
        # or a bare socket path); otherwise auto-detected: rootless user socket first, then rootful /run/podman/podman.sock.
        # Note that rootless Podman cannot expose host GPU/X11 devices to privileged containers; users needing GPUs/GUIs
        # should run from a root-privileged context with the rootful socket enabled (systemctl start podman.socket)
        try:
            from podman import PodmanClient
        except ImportError as e:
            raise RuntimeError("container_engine='podman' requires the Podman Python client: pip install podman") from e
        base_url = os.environ.get("PODMAN_URI") or os.environ.get("PODMAN_HOST")
        if not base_url:
            candidates = []
            getuid = getattr(os, "getuid", None)
            user_sock = f"/run/user/{getuid()}/podman/podman.sock" if getuid else None
            if user_sock and os.path.exists(user_sock):
                candidates.append(user_sock)
            root_sock = "/run/podman/podman.sock"
            if os.path.exists(root_sock):
                candidates.append(root_sock)
            base_url = candidates[0] if candidates else root_sock
        if not base_url.startswith(("http://", "https://", "http+unix://", "http+ssh://", "tcp://", "ssh://")):
            base_url = "http+unix://" + base_url # Accept a bare socket path
        print(f"Connecting to Podman API at {base_url}...")
        client = PodmanClient(base_url=base_url)
        try:
            if not client.ping():
                raise ConnectionError("Podman API replied negatively to ping")
        except Exception as e:
            raise RuntimeError(
                f"Could not reach the Podman API at {base_url}. Enable its Docker-compatible socket "
                "('systemctl --user start podman.socket' for rootless, or 'systemctl start podman.socket' for rootful, "
                "which is recommended with this stack's privileged GPU/X11 containers), or set PODMAN_URI."
            ) from e
        return client

    def _gpu_container_config(self, gpu: bool):
        # Returns (device_requests, device_binds, extra_environment) for the selected engine and vendor.
        # NVIDIA-on-Docker uses the DeviceRequest API ('--gpus all'). AMD GPUs cannot use that anywhere and are
        # passed through as host devices. Podman has no DeviceRequest support at all: NVIDIA cards go through a
        # CDI device instead (requires 'sudo nvidia-ctk cdi generate' from the NVIDIA Container Toolkit).
        # When no usable GPU is found (or for AMD under WSL) nothing is passed through and Gazebo uses software GL.
        if not gpu or self.GPU_VENDOR == "none":
            return [], [], {}
        vendor = self.GPU_VENDOR
        if vendor == "auto":
            if self._nvidia_available():
                vendor = "nvidia"
            elif os.path.exists("/dev/kfd"): # The ROCm kernel driver's compute node is AMD-specific
                vendor = "amd"
            else:
                print("Warning: no NVIDIA (/dev/nvidia*) or AMD (/dev/kfd) GPU detected; running without GPU "
                      "passthrough (Gazebo falls back to software GL). Pass gpu_vendor='nvidia' or 'amd' to force.")
                return [], [], {}
        self.RESOLVED_GPU_VENDOR = vendor
        if vendor == "amd":
            if self._is_wsl():
                # WSL has no /dev/kfd, so AMD GPUs cannot be passed through to containers under WSL
                print("Warning: running under WSL where /dev/kfd does not exist; skipping AMD GPU passthrough.")
                return [], [], {}
            # /dev/kfd (ROCm compute) + /dev/dri (DRM graphics); privileged containers bypass device ACLs anyway,
            # binding them explicitly documents intent and also keeps working in less-permissive setups
            return [], ["/dev/kfd:/dev/kfd:rwm", "/dev/dri:/dev/dri:rwm"], {}
        if self.CONTAINER_ENGINE == "podman":
            return [], ["nvidia.com/gpu=all", "/dev/dri:/dev/dri:rwm"], {} # CDI device, see note above
        device_requests = [
            DeviceRequest(count=-1, capabilities=[['gpu']]) # Replaces "--gpus all"
        ]
        extra_env = {"NVIDIA_DRIVER_CAPABILITIES": "all",
                     "__NV_PRIME_RENDER_OFFLOAD": "1",
                     "__GLX_VENDOR_LIBRARY_NAME": "nvidia"}
        return device_requests, ["/dev/dri:/dev/dri:rwm"], extra_env

    @staticmethod
    def _is_wsl():
        # WSL2 kernel releases look like "5.15.x-microsoft-standard-WSL2"; fall back to /proc/version
        try:
            release = os.uname().release.lower()
            return "microsoft" in release or "wsl" in release
        except AttributeError:
            pass
        try:
            with open("/proc/version", "r") as f:
                return "microsoft" in f.read().lower()
        except OSError:
            return False

    @staticmethod
    def _nvidia_available():
        # A usable NVIDIA stack needs device nodes or a working 'nvidia-smi -L'; the binary alone can be present
        # from an installed-but-broken WSL driver (e.g. NVML fails to initialize with no /dev/nvidia* nodes)
        if os.path.exists("/dev/nvidia0"):
            return True
        if not shutil.which("nvidia-smi"):
            return False
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
            return out.returncode == 0 and bool(out.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            return False

    @staticmethod
    def _gz_model_names(autopilot: str, num_quads: int) -> list:
        # Must mirror the model naming of simulation/simulation.yml.erb (gym_models there):
        # PX4 indexes models by 0-based instance, ArduPilot/Betaflight by 1-based drone_id
        bases = {"px4": "x500", "ardupilot": "iris_with_ardupilot", "betaflight": "iris_with_betaflight"}
        if autopilot not in bases:
            raise ValueError(f"No Gazebo model name mapping for autopilot '{autopilot}'")
        idx_offset = 0 if autopilot == "px4" else 1
        return [f"{bases[autopilot]}_{i + idx_offset}" for i in range(num_quads)]

    def _unpack_reply(self, reply_bytes):
        # Clock (int32 sec, uint32 nanosec) followed by each configured model's ground-truth pose
        # (x, y, z [m] and quaternion w, x, y, z) as doubles, matching zeromq_bridge.cpp
        expected_len = struct.calcsize(self.reply_fmt)
        if len(reply_bytes) != expected_len:
            raise RuntimeError(
                f"Unexpected ZMQ payload size: got {len(reply_bytes)} bytes, "
                f"expected {expected_len} for {len(self.gz_model_names)} model(s). "
                "Ensure the simulation image was rebuilt with the gym_models zmq_bridge parameter."
            )
        unpacked = struct.unpack(self.reply_fmt, reply_bytes)
        sec, nanosec = unpacked[0], unpacked[1]
        raw_poses = np.array(unpacked[2:], dtype=np.float64).reshape(-1, 7)
        if not np.all(np.isfinite(raw_poses)):
            missing = [self.gz_model_names[i] for i in range(len(raw_poses)) if not np.all(np.isfinite(raw_poses[i]))]
            raise RuntimeError(
                f"Ground-truth poses not available for model(s) {missing} from the ZMQ bridge. "
                "Check that zmq_bridge_node logged a successful subscription to /dynamic_pose/info."
            )
        return sec, nanosec, raw_poses[:, :3]

    def _update_from_payload(self, sec, nanosec, pos_all):
        # Store the latest ground truth and estimate own velocity from pose deltas over exact sim time
        self.sim_sec, self.sim_nanosec = sec, nanosec
        sim_time = float(sec) + float(nanosec) * 1e-9
        self.pos_all = pos_all
        own_pos = pos_all[0]
        nominal_dt = 1.0 / self.GYM_FREQ_HZ
        dt = None
        if self.prev_pos is not None and self.prev_sim_time is not None:
            dt_candidate = sim_time - self.prev_sim_time
            if 0.1 * nominal_dt <= dt_candidate <= 10.0 * nominal_dt:
                dt = dt_candidate
                self.vel_est = (own_pos - self.prev_pos) / dt
        if dt is None:
            dt = nominal_dt # First observation of an episode, or implausible clock delta
        self.last_dt = dt
        self.prev_pos = own_pos.copy()
        self.prev_sim_time = sim_time

    def _sample_goal(self):
        # Random goal on an annulus around the spawn point, at roughly the current altitude.
        # Use the np_random property (not the _np_random attribute): Env.reset only sets it when an
        # explicit seed is passed; the property lazily creates an unseeded RNG otherwise
        rng = self.np_random
        radius = rng.uniform(*self.goal_range)
        bearing = rng.uniform(0.0, 2.0 * np.pi)
        z_start = float(self.pos_all[0][2])
        z_goal = min(max(z_start + rng.uniform(-3.0, 3.0), 2.0), 80.0)
        self.goal = self.spawn_xy + radius * np.array([np.cos(bearing), np.sin(bearing)], dtype=np.float64)
        self.goal = np.array([self.goal[0], self.goal[1], z_goal], dtype=np.float64)

    def _get_obs(self):
        own_pos = self.pos_all[0]
        goal_rel = self.goal - own_pos
        obs_parts = [own_pos, self.vel_est, goal_rel]
        for j in range(1, len(self.pos_all)):
            obs_parts.append(self.pos_all[j] - own_pos)  # Other drones, relative to own position
        return np.concatenate(obs_parts).astype(np.float32)

    def _get_info(self):
        own_pos = self.pos_all[0]
        goal_rel = self.goal - own_pos
        horizontal_distance = float(np.linalg.norm(goal_rel[:2]))
        return {
            "sim_time_sec": self.sim_sec,
            "sim_time_nanosec": self.sim_nanosec,
            "position": own_pos.copy(),
            "velocity": self.vel_est.copy(),
            "goal": self.goal.copy(),
            "horizontal_distance_to_goal": horizontal_distance,
            "vertical_error_to_goal": float(goal_rel[2]),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Handle seeding

        # Close existing ZMQ connection if any
        if self.socket:
            self.socket.close()
        # Restart Docker containers
        try:
            print("Restarting all containers in parallel...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.simulation_container.restart, timeout=1)]
                for air_cont in self.aircraft_containers:
                    futures.append(executor.submit(air_cont.restart, timeout=1))
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        except Exception as e:
            print(f"Error restarting containers: {e}")
            raise e
        # Establish ZeroMQ connection
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.REQ_RELAXED, 1) # Allow sending a new request if the previous one timed out
        self.socket.setsockopt(zmq.REQ_CORRELATE, 1) # Match replies to requests to avoid processing late replies
        self.socket.setsockopt(zmq.RCVTIMEO, 60 * 1000) # 60s, only a placeholder, will be changed below
        if self.ZMQ_TRANSPORT == "tcp":
            self.socket.connect(f"tcp://{self.ZMQ_IP}:{self.ZMQ_PORT}")
            print(f"ZeroMQ socket connected to {self.ZMQ_IP}:{self.ZMQ_PORT}")
        elif self.ZMQ_TRANSPORT == "ipc":
            ipc_file = f"{self.ZMQ_IPC_SOCKET_DIR}/bridge_inst{self.INSTANCE}.ipc"
            self.socket.connect(f"ipc://{ipc_file}")
            print(f"ZeroMQ socket connected via IPC: {ipc_file}")
        else:
            raise ValueError(f"Invalid ZMQ_TRANSPORT: {self.ZMQ_TRANSPORT}")
        ###########################################################################################
        # ZeroMQ REQ/REP to the ROS2 sim ##########################################################
        ###########################################################################################
        try:
            self.socket.setsockopt(zmq.RCVTIMEO, 300 * 1000) # Temporarily increase timeout to 300s to reset the simulation
            reset = 9999.0 # A special action to reset the environment
            action_payload = struct.pack('d', reset) # Serialize the action 
            for attempt in range(3):
                try:
                    self.socket.send(action_payload) # Send the REQ
                    reply_bytes = self.socket.recv() # Wait for the REP (synchronous block) this call will block until a reply is received or it times out
                    sec, nanosec, pos_all = self._unpack_reply(reply_bytes)
                    if sec == 0 and nanosec == 0:
                        raise ValueError("Received zero-time payload (C++ bridge timeout).")
                    self.prev_pos = None  # Forget any stale state from the previous episode
                    self.prev_sim_time = None
                    self.vel_est = np.zeros(3, dtype=np.float64)
                    self._update_from_payload(sec, nanosec, pos_all)
                    self.start_sim_sec = float(sec) + (float(nanosec) * 1e-9)
                    break # Success
                except zmq.error.Again:
                    print(f"ZMQ Error: Reply from container timed out. Attempt {attempt + 1} of 3.")
                    if attempt == 2:
                        raise RuntimeError("Simulation reset failed: ZMQ timeout.") from None
                except (ValueError, struct.error):
                    print(f"ZMQ Error: Reply format error. Received garbage state. Attempt {attempt + 1} of 3.")
                    if attempt == 2:
                        raise RuntimeError("Simulation reset failed: Invalid data format.") from None
        finally:
            self.socket.setsockopt(zmq.RCVTIMEO, 60 * 1000) # Restore standard timeout (60s) for stepping
        ###########################################################################################
        ###########################################################################################
        ###########################################################################################
        # The drone has just completed its unpaused init phase (takeoff included): record the spawn
        # reference and sample this episode's random goal around it (goal sampling is seeded by reset(seed=...))
        self.spawn_xy = self.pos_all[0][:2].copy()
        self._sample_goal()
        self.step_count = 0
        self.last_event = None
        
        if self.render_mode == "ansi":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        cmd = np.clip(action, -1.0, 1.0)
        # Scale the normalized action to [vE, vN, vUp (m/s), yaw rate (rad/s)] in the world frame
        vE = float(cmd[0]) * self.max_speed
        vN = float(cmd[1]) * self.max_speed
        vU = float(cmd[2]) * min(self.max_speed, 3.0)
        yaw_rate = float(cmd[3]) * self.max_yaw_rate
        ###########################################################################################
        # ZeroMQ REQ/REP to the ROS2 sim ##########################################################
        ###########################################################################################
        action_payload = struct.pack('<ddddd', 0.0, vE, vN, vU, yaw_rate) # Serialize the step command
        for attempt in range(3):
            try:
                self.socket.send(action_payload) # Send the REQ
                reply_bytes = self.socket.recv() # Wait for the REP (synchronous block) this call will block until a reply is received or it times out
                sec, nanosec, pos_all = self._unpack_reply(reply_bytes)
                if sec == 0 and nanosec == 0:
                    raise ValueError("Received zero-time payload (C++ bridge timeout).")
                break # Success
            except zmq.error.Again:
                print(f"ZMQ Error: Reply from container timed out. Attempt {attempt + 1} of 3.")
                if attempt == 2:
                    raise RuntimeError("Simulation step failed: ZMQ timeout.") from None
            except (ValueError, struct.error):
                print(f"ZMQ Error: Reply format error. Received garbage state. Attempt {attempt + 1} of 3.")
                if attempt == 2:
                    raise RuntimeError("Simulation step failed: Invalid data format.") from None
        ###########################################################################################
        ###########################################################################################
        ###########################################################################################
        # Reward shaping on progress toward the goal, evaluated between the previous and new states
        # (both distances must use the same 3D metric for the potential shaping to telescope correctly)
        d_prev = float(np.linalg.norm(self.goal - self.pos_all[0]))
        self._update_from_payload(sec, nanosec, pos_all)
        own_pos = self.pos_all[0]
        d_cur_3d = np.linalg.norm(self.goal - own_pos)
        d_cur_2d = float(np.linalg.norm(self.goal[:2] - own_pos[:2]))

        reward = self.PROGRESS_GAIN * (d_prev - float(d_cur_3d)) + self.ALIVE_PENALTY_PER_SEC * self.last_dt
        terminated = False

        # Success: inside the goal radius
        if d_cur_2d <= self.GOAL_RADIUS_M and abs(float(self.goal[2] - own_pos[2])) <= self.GOAL_ALT_TOL_M:
            reward += self.SUCCESS_REWARD
            terminated = True
            self.last_event = "success"
        # Crash or landing on the ground: faster contact costs more than a soft touchdown
        elif own_pos[2] <= self.CRASH_AGL_M:
            speed = float(np.linalg.norm(self.vel_est))
            self.last_event = "crash" if speed > 3.0 else "soft_land"
            reward += self.CRASH_REWARD if speed > 3.0 else self.SOFT_LAND_REWARD
            terminated = True
        # Out-of-bounds flyaway
        elif float(np.linalg.norm(own_pos[:2] - self.spawn_xy)) >= self.bounds_radius_m:
            reward += self.OUT_OF_BOUNDS_REWARD
            terminated = True
            self.last_event = "out_of_bounds"

        self.step_count += 1
        # Check for truncation (episode ends due to time limit)
        truncated = self.step_count >= self.max_steps
        if not terminated and truncated:
            self.last_event = "timeout"
        # Get obs and info
        obs = self._get_obs()
        info = self._get_info()
        
        # Handle rendering
        if self.render_mode == "ansi":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            self._render_frame()

    def _render_frame(self):
        bar_width = 40        
        current_abs_time = self.sim_sec + (self.sim_nanosec * 1e-9)
        start_time = getattr(self, 'start_sim_sec', 0.0)
        episode_time = current_abs_time - start_time
        progress = min(max(episode_time / self.MAX_EPISODE_LENGTH_SEC, 0.0), 1.0)
        filled_len = int(bar_width * progress)
        bar = '=' * filled_len + '-' * (bar_width - filled_len)
        print(f"\r[{bar}] {episode_time:6.2f}s / {self.MAX_EPISODE_LENGTH_SEC:.0f}s", end="")

    def close(self):
        if self.render_mode == "ansi":
            print() # Add a newline after the final render
        
        try:
            self.simulation_container.stop(timeout=1)
            self.simulation_container.remove(force=True)
            print(f"Simulation container '{self.simulation_container.name}' stopped.")
        except Exception:
            pass
        for container in self.aircraft_containers:
            try:
                container.stop(timeout=1)
                container.remove(force=True)
                print(f"Aircraft container '{container.name}' stopped.")
            except Exception:
                pass
        for net_name, network_obj in self.networks.items():
            try:
                network_obj.remove()
                print(f"Network {net_name} removed.")
            except Exception as e:
                print(f"Warning: Could not remove {net_name}: {e}")

        # Close ZMQ resources
        if self.socket:
            self.socket.close(linger=0)
        if self.zmq_context:
            self.zmq_context.term()
