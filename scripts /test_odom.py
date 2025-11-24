#!/usr/bin/env python3
"""
测试脚本：订阅并打印里程计信息 (使用标准 nav_msgs/Odometry 消息)
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import numpy as np


class OdomTestNode(Node):
    def __init__(self):
        super().__init__('odom_test_node')
        self.subscription = self.create_subscription(
            Odometry,
            '/robot_odom',
            self.odom_callback,
            10
        )
        self.get_logger().info("里程计测试节点已启动，等待消息...")

    def odom_callback(self, msg: Odometry):
        """打印里程计信息"""
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("里程计信息 (nav_msgs/Odometry):")
        self.get_logger().info(f"  时间戳: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        self.get_logger().info(f"  Frame ID: {msg.header.frame_id} -> {msg.child_frame_id}")

        # 1. 位置信息 (世界坐标系)
        self.get_logger().info("\n1. 位置 (世界坐标系, m):")
        self.get_logger().info(f"   x={msg.pose.pose.position.x:.4f}")
        self.get_logger().info(f"   y={msg.pose.pose.position.y:.4f}")
        self.get_logger().info(f"   z={msg.pose.pose.position.z:.4f}")

        # 2. 四元数信息
        quat = msg.pose.pose.orientation
        self.get_logger().info("\n2. 四元数 (归一化检查):")
        self.get_logger().info(f"   w={quat.w:.4f}, x={quat.x:.4f}, y={quat.y:.4f}, z={quat.z:.4f}")
        norm = np.sqrt(quat.w ** 2 + quat.x ** 2 + quat.y ** 2 + quat.z ** 2)
        self.get_logger().info(f"   归一化: {norm:.6f} (应为1.0)")

        # 3. 欧拉角 (从四元数转换)
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        euler = r.as_euler('xyz', degrees=False)
        self.get_logger().info("\n3. 欧拉角 (世界坐标系, rad):")
        self.get_logger().info(f"   roll={euler[0]:.4f}, pitch={euler[1]:.4f}, yaw={euler[2]:.4f}")

        # 4. 线速度信息 (机器人坐标系)
        self.get_logger().info("\n4. 线速度 (机器人坐标系, m/s):")
        self.get_logger().info(f"   vx={msg.twist.twist.linear.x:.4f}")
        self.get_logger().info(f"   vy={msg.twist.twist.linear.y:.4f}")
        self.get_logger().info(f"   vz={msg.twist.twist.linear.z:.4f}")

        # 5. 角速度信息 (机器人坐标系)
        self.get_logger().info("\n5. 角速度 (机器人坐标系, rad/s):")
        self.get_logger().info(f"   wx={msg.twist.twist.angular.x:.4f} (roll rate)")
        self.get_logger().info(f"   wy={msg.twist.twist.angular.y:.4f} (pitch rate)")
        self.get_logger().info(f"   wz={msg.twist.twist.angular.z:.4f} (yaw rate)")

        self.get_logger().info("=" * 60 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = OdomTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()