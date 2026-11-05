#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <array>
#include <map>
#include <sstream>
#include <limits>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
// ROS 2 Includes
#include "rclcpp/rclcpp.hpp"
#include "rosgraph_msgs/msg/clock.hpp"
// Gazebo Includes (libgz-transport13-*, libgz-msgs10-dev)
#include <gz/transport/Node.hh>
#include <gz/msgs/world_control.pb.h>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/pose_v.pb.h>
// ZeroMQ Includes (requires cppzmq installed with libzmq3-dev)
#include <zmq.hpp> 

using namespace std::chrono_literals;

// Corresponds to Python struct.unpack('iI' + 'd'*7*num_models)
#pragma pack(push, 1)
struct ClockPayload {
    int32_t sec;
    uint32_t nanosec;
};
// Ground-truth pose of one model, in the Gazebo world frame (x, y, z [m], then quaternion w, x, y, z)
struct PosePayload {
    double xyzw[7];
};
#pragma pack(pop)

class ZMQBridge : public rclcpp::Node {
public:
    ZMQBridge() : Node("zmq_bridge_node"), context_(1), socket_(context_, zmq::socket_type::rep) {

        this->declare_parameter("step_size", 250); // How many multiples of the timestep in the world SDF (250Hz/4ms for PX4, 500Hz/2ms for ArduPilot)
        step_size_ = this->get_parameter("step_size").as_int();
        this->declare_parameter("physics_dt", 0.004); // Size of the physics timestep in seconds (4ms for PX4, 2ms for ArduPilot)
        physics_dt_ = this->get_parameter("physics_dt").as_double();
        RCLCPP_INFO(this->get_logger(), "Config: step size = %d, physics dt = %.4f", step_size_, physics_dt_);
        this->declare_parameter("init_duration", 80.0); // Duration to run unpaused during reset (seconds)
        init_duration_ = this->get_parameter("init_duration").as_double();

        // RL observation models: comma-separated list of Gazebo model names whose ground-truth poses are
        // appended to every REP payload, in order. Empty string keeps the legacy clock-only payload.
        std::string gym_models = this->declare_parameter("gym_models", ""); // e.g. "iris_with_ardupilot_1,iris_with_ardupilot_2"
        std::stringstream model_stream(gym_models);
        std::string model_name;
        while (std::getline(model_stream, model_name, ',')) {
            if (!model_name.empty()) gym_model_names_.push_back(model_name);
        }
        if (!gym_model_names_.empty()) RCLCPP_INFO(this->get_logger(), "Gym RL observation models: %s", gym_models.c_str());

        // RL action relay: plain UDP endpoint ("ip:port") that receives the per-step velocity command
        // '<ffff' (vE, vN, vUp [m/s], yaw rate [rad/s]) so an in-container node can feed it to the SITL.
        std::string gym_udp_target = this->declare_parameter("gym_udp_target", ""); // e.g. "10.42.90.1:5871"
        if (!gym_udp_target.empty()) {
            size_t colon = gym_udp_target.rfind(':');
            if (colon == std::string::npos) {
                RCLCPP_ERROR(this->get_logger(), "Invalid gym_udp_target '%s' (expected ip:port); RL actions disabled.", gym_udp_target.c_str());
            } else {
                memset(&relay_addr_, 0, sizeof(relay_addr_));
                relay_addr_.sin_family = AF_INET;
                relay_addr_.sin_port = htons(static_cast<uint16_t>(std::stoi(gym_udp_target.substr(colon + 1))));
                if (inet_pton(AF_INET, gym_udp_target.substr(0, colon).c_str(), &relay_addr_.sin_addr) != 1) {
                    RCLCPP_ERROR(this->get_logger(), "Invalid gym_udp_target IP in '%s'; RL actions disabled.", gym_udp_target.c_str());
                } else {
                    relay_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
                    if (relay_fd_ < 0) {
                        RCLCPP_ERROR(this->get_logger(), "Could not create UDP relay socket; RL actions disabled.");
                    } else {
                        RCLCPP_INFO(this->get_logger(), "Relaying Gym RL actions via UDP to %s", gym_udp_target.c_str());
                    }
                }
            }
        }
        
        // // 1. Option A: ZMQ TCP Setup (see self.ZMQ_TRANSPORT in aas_env.py)
        // socket_.bind("tcp://*:5555"); // '*' binds to all interfaces (equivalent to 0.0.0.0)
        // RCLCPP_INFO(this->get_logger(), "ZMQ REP socket bound to port 5555");

        // 1. Option B: ZMQ IPC Setup (see self.ZMQ_TRANSPORT in aas_env.py)
        const char* env_instance = std::getenv("INSTANCE");
        if (env_instance == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "Environment variable 'INSTANCE' was not set.");
        }
        std::string instance_id = env_instance;
        std::string ipc_address = "ipc:///tmp/aas_zmq_sockets/bridge_inst" + instance_id + ".ipc";
        try {
            socket_.bind(ipc_address);
            std::string socket_path = ipc_address.substr(6); // Set permissions to 777 (anyone can read/write) so that aas_env can connect
            chmod(socket_path.c_str(), 0777);
            RCLCPP_INFO(this->get_logger(), "ZMQ REP socket bound to: %s", ipc_address.c_str());
        } catch (zmq::error_t& e) {
            RCLCPP_ERROR(this->get_logger(), "ZMQ Bind Error: %s", e.what());
        }

        // 2. ROS 2 Setup
        subscription_ = this->create_subscription<rosgraph_msgs::msg::Clock>(
            "/clock", 10, 
            std::bind(&ZMQBridge::clock_callback, this, std::placeholders::_1));

        const char* env_world = std::getenv("WORLD");
        if (env_world == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "Environment variable 'WORLD' was not set.");
        }
        std::string world_name = env_world;
        service_topic_ = "/world/" + world_name + "/control";

        subscribe_poses(world_name);

        // 3. Start ZMQ Thread
        running_ = true;
        zmq_thread_ = std::thread(&ZMQBridge::zmq_listener, this);
    }

    ~ZMQBridge() {
        running_ = false;
        clock_cv_.notify_all(); // Wake up the ZMQ thread if waiting for the clock
        if (zmq_thread_.joinable()) {
            zmq_thread_.join();
        }
        // Clean shutdown for ZMQ and the RL action relay
        context_.close();
        if (relay_fd_ >= 0) {
            close(relay_fd_);
            relay_fd_ = -1;
        }
    }

private:
    zmq::context_t context_;
    zmq::socket_t socket_;
    
    rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr subscription_;
    
    gz::transport::Node gz_node_;
    std::string service_topic_;

    std::thread zmq_thread_;
    std::atomic<bool> running_;

    std::mutex clock_mutex_;
    std::condition_variable clock_cv_;
    bool clock_ready_ = false;
    ClockPayload current_clock_;
    double current_sim_time_ = 0.0;

    int step_size_;
    double physics_dt_;
    double init_duration_;

    std::vector<std::string> gym_model_names_;
    std::map<std::string, PosePayload> latest_pose_; // Guarded by pose_mutex_
    std::mutex pose_mutex_;
    int relay_fd_ = -1;
    struct sockaddr_in relay_addr_;

    void clock_callback(const rosgraph_msgs::msg::Clock::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(clock_mutex_);
        
        current_clock_.sec = msg->clock.sec;
        current_clock_.nanosec = msg->clock.nanosec;
        current_sim_time_ = msg->clock.sec + (msg->clock.nanosec * 1e-9);
        
        clock_ready_ = true;
        clock_cv_.notify_all(); // Notify waiting threads
    }

    void set_gazebo_pause(bool pause) {
        gz::msgs::WorldControl req;
        req.set_pause(pause);
        gz::msgs::Boolean rep;
        bool result;
        gz_node_.Request(service_topic_, req, 1000, rep, result);
    }

    void step_gazebo() {
        gz::msgs::WorldControl req;
        req.set_pause(true);
        req.set_multi_step(static_cast<unsigned int>(step_size_));
        gz::msgs::Boolean rep;
        bool result;
        gz_node_.Request(service_topic_, req, 1000, rep, result);
    }

    void subscribe_poses(const std::string& world_name) {
        if (gym_model_names_.empty()) return;
        std::string pose_topic = "/world/" + world_name + "/dynamic_pose/info";
        bool ok = gz_node_.Subscribe(pose_topic, 10, &ZMQBridge::pose_callback, this);
        if (ok) {
            RCLCPP_INFO(this->get_logger(), "Subscribed to ground-truth poses on %s", pose_topic.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Could not subscribe to %s; RL observations will be zeros.", pose_topic.c_str());
        }
    }

    void pose_callback(const gz::msgs::Pose_V& msg) {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        for (const auto& pose : msg.pose()) {
            const std::string name = pose.name(); // Populated by Gazebo with the model name
            for (const auto& wanted : gym_model_names_) {
                if (name == wanted) {
                    PosePayload p;
                    p.xyzw[0] = pose.position().x();
                    p.xyzw[1] = pose.position().y();
                    p.xyzw[2] = pose.position().z();
                    p.xyzw[3] = pose.orientation().w();
                    p.xyzw[4] = pose.orientation().x();
                    p.xyzw[5] = pose.orientation().y();
                    p.xyzw[6] = pose.orientation().z();
                    latest_pose_[name] = p;
                }
            }
        }
    }

    void zmq_listener() {
        RCLCPP_INFO(this->get_logger(), "ZMQ listener thread started.");
        
        zmq::pollitem_t items[] = {
            { static_cast<void*>(socket_), 0, ZMQ_POLLIN, 0 }
        };

        while (rclcpp::ok() && running_) {
            zmq::poll(&items[0], 1, 100); // Poll with 100ms timeout to allow checking running_ flag

            if (items[0].revents & ZMQ_POLLIN) {
                try {
                    // 1. Receive Action: either a single reset marker double (legacy-compatible),
                    //    or a step payload [marker, vE, vN, vUp, yaw_rate] of doubles
                    zmq::message_t request;
                    auto res = socket_.recv(request, zmq::recv_flags::none);
                    if (!res) continue;
                    const size_t num_doubles = request.size() / sizeof(double);
                    const double* payload_doubles = static_cast<const double*>(request.data());
                    const bool is_reset = num_doubles >= 1 && std::abs(payload_doubles[0] - 9999.0) < 0.001;

                    bool received =  true; // Initialize and default to true

                    // 2. Logic Dispatch
                    if (is_reset) { // RESET MODE
                        RCLCPP_INFO(this->get_logger(), "Received reset action (9999.0). Running unpaused...");
                        {
                            std::lock_guard<std::mutex> lock(pose_mutex_);
                            latest_pose_.clear(); // Force the fresh post-respawn poses to be collected before replying
                        }

                        // A. Unpause simulation
                        double reset_start;
                        {
                            std::lock_guard<std::mutex> lock(clock_mutex_);
                            reset_start = current_sim_time_;
                        }
                        set_gazebo_pause(false);
                        // B. Wait loop
                        while (rclcpp::ok() && running_) {
                            std::unique_lock<std::mutex> lock(clock_mutex_);
                            if (clock_cv_.wait_for(lock, 200ms, [this, reset_start]{
                                return current_sim_time_ >= (reset_start + init_duration_);
                            })) {
                                break;
                            }
                        }
                        // C. Pause simulation
                        set_gazebo_pause(true);
                        RCLCPP_INFO(this->get_logger(), "Initialization Complete. Paused.");

                    } else { // STEP MODE
                        // A. Relay the RL action (vE, vN, vUp [m/s], yaw rate [rad/s]) to the SITL-connected node,
                        //    before stepping Gazebo so that this step is flown with the new setpoint
                        if (relay_fd_ >= 0 && num_doubles >= 5) {
                            float cmd[4];
                            for (int i = 0; i < 4; i++) cmd[i] = static_cast<float>(payload_doubles[1 + i]);
                            ssize_t sent = sendto(relay_fd_, cmd, sizeof(cmd), 0,
                                                  reinterpret_cast<const struct sockaddr*>(&relay_addr_), sizeof(relay_addr_));
                            if (sent < 0) {
                                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "UDP RL action relay send failed: %s", strerror(errno));
                            }
                        }
                        // B. Calculate Target Time
                        double target_time;
                        {
                            std::lock_guard<std::mutex> lock(clock_mutex_);
                            target_time = current_sim_time_ + (step_size_ * physics_dt_) - 0.0001;
                            clock_ready_ = false;
                        }
                        // C. Trigger Step
                        step_gazebo();
                        // D. Wait for Result
                        std::unique_lock<std::mutex> lock(clock_mutex_);
                        received = clock_cv_.wait_for(lock, 30000ms, [this, target_time]{
                            return current_sim_time_ >= target_time;
                        }); // Wait up to 30 seconds for clock_ready_ to become true
                    }

                    // 3. Send Reply: clock (+ ground-truth poses of gym_model_names_, in configured order)
                    const size_t reply_size = sizeof(ClockPayload) + gym_model_names_.size() * sizeof(PosePayload);
                    zmq::message_t reply(reply_size);
                    ClockPayload snapshot;
                    {
                        std::lock_guard<std::mutex> lock(clock_mutex_);
                        snapshot = current_clock_;
                    }
                    if (!received) {
                        RCLCPP_WARN(this->get_logger(), "Update timeout!");
                        snapshot = {0, 0};
                    }
                    std::memcpy(reply.data(), &snapshot, sizeof(ClockPayload));
                    if (!gym_model_names_.empty()) {
                        std::lock_guard<std::mutex> lock(pose_mutex_);
                        size_t offset = sizeof(ClockPayload);
                        for (const auto& name : gym_model_names_) {
                            PosePayload pose_nan;
                            const double nan = std::numeric_limits<double>::quiet_NaN();
                            for (int i = 0; i < 7; i++) pose_nan.xyzw[i] = nan;
                            auto it = latest_pose_.find(name);
                            if (it != latest_pose_.end()) pose_nan = it->second;
                            std::memcpy(static_cast<char*>(reply.data()) + offset, &pose_nan, sizeof(PosePayload));
                            offset += sizeof(PosePayload);
                        }
                    }
                    socket_.send(reply, zmq::send_flags::none);

                } catch (const std::exception &e) {
                    RCLCPP_ERROR(this->get_logger(), "ZMQ Error: %s", e.what());
                }
            }
        }
    }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ZMQBridge>());
    rclcpp::shutdown();
    return 0;
}
