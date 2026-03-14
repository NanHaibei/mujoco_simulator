from sensor_msgs.msg import JointState

from .base_plugin import BasePlugin


class JointStatePlugin(BasePlugin):
    """关节状态发布插件
    
    负责发布关节状态用于可视化（如 rviz）。
    """
    
    def init(self):
        """初始化关节状态插件"""
        # 创建发布者
        self.joint_state_pub = self.simulator.create_publisher(
            JointState, "/joint_states", 10
        )
        
        # 设置定时器（60Hz发布）
        self.update_rate = 60.0
        self.simulator.create_timer(1.0 / self.update_rate, self._publish_joint_states)
    
    def _publish_joint_states(self):
        """发布关节状态"""
        if self.simulator.read_error_flag:
            return
        
        joint_state = JointState()
        joint_state.header.stamp = self.simulator.get_clock().now().to_msg()
        joint_state.name = self.simulator.joint_name.copy()
        joint_state.position = self.simulator.sensor_data_list[
            self.simulator.joint_pos_head_id : self.simulator.joint_pos_head_id + self.mj_model.nu
        ]
        joint_state.velocity = self.simulator.sensor_data_list[
            self.simulator.joint_vel_head_id : self.simulator.joint_vel_head_id + self.mj_model.nu
        ]
        joint_state.effort = self.simulator.sensor_data_list[
            self.simulator.joint_tor_head_id : self.simulator.joint_tor_head_id + self.mj_model.nu
        ]
        self.joint_state_pub.publish(joint_state)
    
    def execute(self):
        """执行函数 - 由定时器调用，不需要在这里实现"""
        pass