#!/usr/bin/env python3
"""
RL action relay for the Gymnasium environment (AUTOPILOT=ardupilot).

Receives per-step velocity commands (sent by zmq_bridge_node inside the simulation container,
which forwards them from the AASEnv.step() payload) on a UDP socket bound to this aircraft
container's own IP, and feeds them to the SITL through MAVROS as ENU velocity setpoints
(/mavros/setpoint_velocity/cmd_vel), switching the vehicle into GUIDED mode.

Safety behavior:
- No action for ACTION_TIMEOUT_SEC while active -> publish zero velocity (hover) and switch back to LOITER.
- Actions resume -> re-enter GUIDED mode automatically.

Run as (one instance per drone, inside the aircraft container; see aircraft.yml.erb):
    python3 /aas/aircraft_resources/scripts/rl_guided_relay.py --drone-id 1 --bind-port 5871 --ros-args -p use_sim_time:=true
"""

import argparse
import socket
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode


class RLGuidedRelay(Node):
    def __init__(self, bind_ip: str, bind_port: int, action_timeout_sec: float = 3.0):
        super().__init__('rl_guided_relay')

        self.ACTION_TIMEOUT_SEC = action_timeout_sec
        self.MODE_RETRY_SEC = 2.0 # Retry interval for the GUIDED/LOITER mode requests while in each state

        # UDP socket receiving '<ffff' (vE, vN, vUp [m/s], yaw rate [rad/s]) from the simulation container
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.bind((bind_ip, bind_port))
        self.udp_socket.setblocking(False)
        self.get_logger().info(f"RL guided relay bound to {bind_ip}:{bind_port}, waiting for Gym actions...")

        # MAVROS interfaces (QoS matches the mavros setpoint_velocity plugin: BEST_EFFORT depth 1)
        from rclpy.qos import QoSPresetProfiles
        self.setpoint_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel',
            QoSPresetProfiles.SENSOR_DATA.value)
        self.state_sub = self.create_subscription(
            State, '/mavros/state', self.state_callback, 10)
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.current_state = State()
        self.last_setpoint_time = None # Wall time of the last received UDP action (None = never active)
        self.relaying = False
        self.last_mode_request_time = 0.0
        self.last_service_warn_time = 0.0

    def state_callback(self, msg: State):
        self.current_state = msg

    def warn_throttled(self, message: str):
        now = time.monotonic()
        if now - self.last_service_warn_time >= 5.0:
            self.get_logger().warn(message)
            self.last_service_warn_time = now

    def request_mode(self, custom_mode: str):
        now = time.monotonic()
        if now - self.last_mode_request_time < self.MODE_RETRY_SEC:
            return
        if not self.set_mode_client.wait_for_service(timeout_sec=0.0):
            self.warn_throttled("MAVROS /mavros/set_mode service not available yet.")
            return
        req = SetMode.Request()
        req.custom_mode = custom_mode
        self.set_mode_client.send_request_async(req, self.mode_response_callback)
        self.last_mode_request_time = now

    def mode_response_callback(self, future):
        try:
            response = future.result()
        except Exception as e: # noqa: BLE001
            self.get_logger().warn(f"Mode request failed: {e}")
            return
        if not response.mode_sent:
            self.get_logger().warn("MAVROS could not parse the requested mode (SET_MODE not sent).")

    def publish_twist(self, vE: float, vN: float, vUp: float, yaw_rate: float):
        twist = TwistStamped()
        # get_clock() honors the use_sim_time parameter so timestamps match the SITL clock in lockstep runs
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = 'enu'
        twist.twist.linear.x = vE
        twist.twist.linear.y = vN
        twist.twist.linear.z = vUp
        twist.twist.angular.z = yaw_rate
        self.setpoint_pub.publish(twist)

    def spin_once(self, timeout_sec: float):
        rclpy.spin_once(self, timeout_sec=timeout_sec)

        now = time.monotonic()

        # Drain UDP action datagrams (only the newest one matters for this control instant)
        latest_cmd = None
        try:
            while True:
                data = self.udp_socket.recv(16)
                if len(data) == 16:
                    latest_cmd = struct.unpack('<ffff', data)
        except BlockingIOError:
            pass
        except OSError as e: # noqa: BLE001
            self.get_logger().error(f"UDP socket error: {e}")

        if latest_cmd is not None:
            vE, vN, vUp, yaw_rate = [min(max(float(v), -30.0), 30.0) for v in latest_cmd]
            self.last_setpoint_time = now
            if not self.relaying:
                self.get_logger().info("Gym action stream active, requesting GUIDED mode.")
                self.relaying = True
            self.request_mode('GUIDED') # Rate-limited; no-op when already in GUIDED
            self.publish_twist(vE, vN, vUp, yaw_rate)

        if self.relaying and self.last_setpoint_time is not None and now - self.last_setpoint_time > self.ACTION_TIMEOUT_SEC:
            self.get_logger().warn("Gym action stream stalled, hovering then handing over to LOITER.")
            # Hover in place for one timeout period before returning control to the autopilot's LOITER
            self.publish_twist(0.0, 0.0, 0.0, 0.0)
            if now - self.last_setpoint_time > 2.0 * self.ACTION_TIMEOUT_SEC:
                self.request_mode('LOITER')
                if self.current_state.mode == 'LOITER':
                    self.relaying = False

    def close(self):
        self.udp_socket.close()


def main():
    parser = argparse.ArgumentParser(description='Gymnasium RL action relay (ArduPilot/MAVLink->MAVROS)')
    parser.add_argument('--drone-id', type=int, required=True, help='Drone ID; binds UDP port 5870 + drone_id on all interfaces')
    parser.add_argument('--bind-ip', type=str, default='0.0.0.0', help='UDP bind address (default: all interfaces)')
    parser.add_argument('--action-timeout-sec', type=float, default=3.0, help='Wall seconds of missing actions before reverting to LOITER')
    parsed_args, ros_args = parser.parse_known_args()

    if not 1 <= parsed_args.drone_id <= 9:
        sys.exit(f"--drone-id must be between 1 and 9 (got {parsed_args.drone_id})")

    bind_port = 5870 + parsed_args.drone_id

    rclpy.init(args=ros_args) # Passes through ROS arguments such as -p use_sim_time:=true
    node = RLGuidedRelay(parsed_args.bind_ip, bind_port, parsed_args.action_timeout_sec)
    try:
        while rclpy.ok():
            node.spin_once(timeout_sec=0.02) # 50 Hz loop ceiling, aligned with AASEnv.GYM_FREQ_HZ defaults
    finally:
        node.close()
        try:
            rclpy.shutdown() # May already be shut down if rclpy.ok() went false externally
        except Exception: # noqa: BLE001
            pass


if __name__ == '__main__':
    main()
