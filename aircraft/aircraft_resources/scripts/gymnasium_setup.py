import argparse

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from autopilot_interface_msgs.action import Takeoff, Offboard
from mavros_msgs.msg import Altitude
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode as MavrosSetMode


class GymnasiumSetup(Node):
    def __init__(self, drone_id, vtol_mc_hold=False):
        super().__init__('gymnasium_setup_node')
        self.drone_id = drone_id
        
        self.vtol_mc_hold = vtol_mc_hold

        self.takeoff_client = ActionClient(self, Takeoff, f'/Drone{drone_id}/takeoff_action')
        self.offboard_client = ActionClient(self, Offboard, f'/Drone{drone_id}/offboard_action')

        if vtol_mc_hold:
            # MAVROS primitives for the VTOL multirotor-phase takeoff (only wired for AUTOPILOT=ardupilot)
            self.mavros_set_mode_client = self.create_client(MavrosSetMode, '/mavros/set_mode')
            self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
            self.takeoff_cmd_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
            self.above_home = None # Altitude (m) above home from the MAVROS altitude plugin
            self.altitude_sub = self.create_subscription(Altitude, '/mavros/altitude', self.altitude_callback, 10)

    def wait_for_server(self, client, name):
        self.get_logger().info(f'Waiting for {name} action server...')
        while not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().info(f'{name} not available yet. Retrying...')
        self.get_logger().info(f'{name} server is ready.')

    def send_takeoff(self):
        self.wait_for_server(self.takeoff_client, 'Takeoff')
        
        goal_msg = Takeoff.Goal()
        goal_msg.takeoff_altitude = 40.0
        goal_msg.vtol_transition_heading = 330.0
        goal_msg.vtol_loiter_north = 100.0
        goal_msg.vtol_loiter_east = 100.0
        goal_msg.vtol_loiter_alt = 60.0

        self.get_logger().info('Sending Takeoff Goal...')
        
        send_goal_future = self.takeoff_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Takeoff Goal Rejected! Retrying...')
            return False

        self.get_logger().info('Takeoff Goal Accepted. Waiting for result...')
        
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        
        return True

    def altitude_callback(self, msg):
        self.above_home = float(msg.above_home)

    def set_mavros_mode(self, mode: str) -> bool:
        if not self.mavros_set_mode_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f'/mavros/set_mode service not available yet for {mode}.')
            return False
        req = MavrosSetMode.Request()
        req.custom_mode = mode
        future = self.mavros_set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if not future.result().mode_sent:
            self.get_logger().error(f'MAVROS could not parse the requested mode {mode} (SET_MODE not sent).')
            return False
        self.get_logger().info(f'Set MAVROS mode to {mode}.')
        return True

    def send_vtol_mc_hold(self, takeoff_altitude=40.0) -> bool:
        # ArduPilot VTOL: first half of the C++ Takeoff action FSM for mav_type==1 (ardupilot_interface.cpp),
        # stopping before the CRUISE/mission stages so the vehicle ends hovering in GUIDED mode's MC phase --
        # the same end state as a quad takeoff, which is what the RL relay takes over from. Each failed step
        # returns False and the outer loop retries; re-issuing already-completed steps is harmless (QLOITER
        # while flying just hovers in place).
        if not self.set_mavros_mode('QLOITER'):
            return False
        spin_wait(self, 2.0)

        if not self.arm_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/mavros/cmd/arming service not available yet.')
            return False
        arm_req = CommandBool.Request()
        arm_req.value = True
        future = self.arm_client.call_async(arm_req)
        rclpy.spin_until_future_complete(self, future)
        if not future.result().success:
            # ArduPilot rejects arming until its preflight checks pass (~30 s after SITL start); the outer loop retries
            self.get_logger().warn('Arming rejected (preflight checks pending?), will retry.')
            return False
        spin_wait(self, 2.0)

        if not self.set_mavros_mode('GUIDED'):
            return False
        spin_wait(self, 2.0)

        if not self.takeoff_cmd_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/mavros/cmd/takeoff service not available yet.')
            return False
        tol_req = CommandTOL.Request()
        tol_req.altitude = takeoff_altitude
        future = self.takeoff_cmd_client.call_async(tol_req)
        rclpy.spin_until_future_complete(self, future)
        if not future.result().success:
            self.get_logger().warn('Takeoff command rejected, will retry.')
            return False

        # Climb to the target altitude (mirrors VTOL_TAKEOFF_ALT_THRESH=2.0 m in ardupilot_interface.cpp),
        # with a sim-time deadline so a stalled climb triggers an outer-loop retry
        deadline = self.get_clock().now() + Duration(seconds=60.0)
        while self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1) # Process /mavros/altitude callbacks
            if self.above_home is not None and abs(self.above_home - takeoff_altitude) < 2.0:
                self.get_logger().info(f'VTOL MC hold reached {takeoff_altitude:.1f} m above home in GUIDED mode.')
                return True
        self.get_logger().warn(f'Did not reach {takeoff_altitude:.1f} m within the climb deadline, will retry.')
        return False

    def send_offboard(self):
        self.wait_for_server(self.offboard_client, 'Offboard')

        goal_msg = Offboard.Goal()
        goal_msg.controller_name = "ctbr-test" # See px4_offboard.cpp
        goal_msg.max_duration_sec = 1200.0 # 20' of offboard mode

        self.get_logger().info('Sending Offboard Goal...')
        send_goal_future = self.offboard_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Offboard Goal Rejected.')
            return False
        
        self.get_logger().info('Offboard Mode Active.')
        return True

def spin_wait(node, seconds):
    target_time = node.get_clock().now() + Duration(seconds=seconds)
    while node.get_clock().now() < target_time:
        rclpy.spin_once(node, timeout_sec=0.1) # Process callbacks to receive /clock updates

def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='Gymnasium Setup Node')
    parser.add_argument('--drone_id', type=str, required=True, help='The ID of the drone')
    parser.add_argument('--vtol-mc-hold', action='store_true',
                        help='ArduPilot VTOL only: take off and hold in GUIDED mode multirotor phase (no FW transition) so RL can fly from a hover')
    parsed_args, _ = parser.parse_known_args()
    drone_id = parsed_args.drone_id

    node = GymnasiumSetup(drone_id, vtol_mc_hold=parsed_args.vtol_mc_hold)

    # Takeoff
    takeoff_success = False
    while not takeoff_success:
        if node.vtol_mc_hold:
            takeoff_success = node.send_vtol_mc_hold()
        else:
            takeoff_success = node.send_takeoff()
        if not takeoff_success:
            spin_wait(node, 2.0) # Simulation time

    # # Offboard
    # offboard_success = False
    # while not offboard_success:
    #     offboard_success = node.send_offboard()
    #     if not offboard_success:
    #         spin_wait(node, 1.0) # Simulation time

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
