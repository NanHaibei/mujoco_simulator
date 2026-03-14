from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

from .base_plugin import BasePlugin


class OdomPlugin(BasePlugin):
    """里程计插件
    
    负责发布机器人里程计信息。
    """
    
    def init(self):
        """初始化里程计插件"""
        # 读取配置
        self.odom_topic = self.simulator.param.get("odomTopic", "/robot_odom")
        
        # 创建发布者
        self.odom_pub = self.simulator.create_publisher(Odometry, self.odom_topic, 10)
    
    def execute(self):
        """执行里程计发布"""
        if not self.enabled:
            return
        
        # 如果模型读取有错误，则不执行操作
        if self.simulator.read_error_flag:
            return
        
        # 创建 Odometry 消息
        odom_msg = Odometry()
        odom_msg.header.stamp = self.simulator.get_clock().now().to_msg()
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = self.simulator.first_link_name
        
        # === Pose 部分 (相对于 world 坐标系) ===
        # 1. 位置信息
        odom_msg.pose.pose.position.x = float(self.simulator.sensor_data_list[self.simulator.real_pos_head_id + 0])
        odom_msg.pose.pose.position.y = float(self.simulator.sensor_data_list[self.simulator.real_pos_head_id + 1])
        odom_msg.pose.pose.position.z = float(self.simulator.sensor_data_list[self.simulator.real_pos_head_id + 2])
        
        # 2. 四元数信息
        odom_msg.pose.pose.orientation.w = float(self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 0])
        odom_msg.pose.pose.orientation.x = float(self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 1])
        odom_msg.pose.pose.orientation.y = float(self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 2])
        odom_msg.pose.pose.orientation.z = float(self.simulator.sensor_data_list[self.simulator.imu_quat_head_id + 3])
        
        # 位置协方差
        odom_msg.pose.covariance = [0.0] * 36
        
        # === Twist 部分 (相对于 child_frame_id) ===
        # 3. 线速度信息
        if self.simulator.real_vel_head_id != 999999:
            odom_msg.twist.twist.linear.x = float(self.simulator.sensor_data_list[self.simulator.real_vel_head_id + 0])
            odom_msg.twist.twist.linear.y = float(self.simulator.sensor_data_list[self.simulator.real_vel_head_id + 1])
            odom_msg.twist.twist.linear.z = float(self.simulator.sensor_data_list[self.simulator.real_vel_head_id + 2])
        else:
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0
        
        # 4. 角速度信息
        odom_msg.twist.twist.angular.x = float(self.simulator.sensor_data_list[self.simulator.imu_gyro_head_id + 0])
        odom_msg.twist.twist.angular.y = float(self.simulator.sensor_data_list[self.simulator.imu_gyro_head_id + 1])
        odom_msg.twist.twist.angular.z = float(self.simulator.sensor_data_list[self.simulator.imu_gyro_head_id + 2])
        
        # 速度协方差
        odom_msg.twist.covariance = [0.0] * 36
        
        # 发布里程计信息
        self.odom_pub.publish(odom_msg)
        
        # 发布 TF 变换: world -> first_link_name
        odom_tf = TransformStamped()
        odom_tf.header.stamp = odom_msg.header.stamp
        odom_tf.header.frame_id = "world"
        odom_tf.child_frame_id = self.simulator.first_link_name
        
        odom_tf.transform.translation.x = odom_msg.pose.pose.position.x
        odom_tf.transform.translation.y = odom_msg.pose.pose.position.y
        odom_tf.transform.translation.z = odom_msg.pose.pose.position.z
        
        odom_tf.transform.rotation.w = odom_msg.pose.pose.orientation.w
        odom_tf.transform.rotation.x = odom_msg.pose.pose.orientation.x
        odom_tf.transform.rotation.y = odom_msg.pose.pose.orientation.y
        odom_tf.transform.rotation.z = odom_msg.pose.pose.orientation.z
        
        self.simulator.tf_broadcaster.sendTransform(odom_tf)