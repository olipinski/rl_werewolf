This dir contains the wrappers for Gym envs.
Wrapper do not contain logic for the game, they extend some functionalities such as parametric actions, metrics ... 

### PaWrapper
[PaWrapper](PaWrapper.py) stands for ParametricActionEnvironment.
It is basically a wrapper around PaEnv to allow custom action masking. 
Together with the original observation space a boolean numpy array is used as an action masking. IndexOf zeros in the
 mask array will be interpreted with non-executable/eatable agents by the model. 

### EvaluationWrapper
[EvaluationWrapper](EvalWrapper.py) stands for Evaluation Environment.
It is built on top of the ParametricActionWrapper, and it uses classes from the [evaluation dir](../../evaluation) to understand what the agents are learning.

#### TODO
- move logs here [X]
- move custom metrics here [X]
- add penalties metrics 