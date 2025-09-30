from collections import deque
from mit_msgs.msg import MITLowState, MITJointCommand, MITJointCommands

low_state_msg = MITLowState()
low_state_msg.joint_states.position = [0.0 for _ in range(22)]
low_state_msg.joint_states.velocity = [0.0 for _ in range(22)]
low_state_msg.joint_states.effort = [0.0 for _ in range(22)]

state_deque = deque()
for _ in range(10):
    state_deque.append(low_state_msg)


new_low_state_msg = MITLowState()
new_low_state_msg.joint_states.position = [float(i) for i in range(22)]
new_low_state_msg.joint_states.velocity = [float(i*2) for i in range(22)]
new_low_state_msg.joint_states.effort = [float(i*3) for i in range(22)]

state_deque.append(new_low_state_msg)
for i in range(len(state_deque)):
    print(state_deque[i].joint_states.position[1])
state_deque.append(new_low_state_msg)
for i in range(len(state_deque)):
    print(state_deque[i].joint_states.position[1])
a = state_deque.popleft()
b = state_deque.popleft()


