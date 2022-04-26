# F3_Net
A neural network about DeepFakes

对《Thinking in Frequency: Face Forgery Detectionby Mining Frequency-aware Clues》一文的尝试复现

关于文件的解释：
- model文件夹中的models.py 是原论文中含有FAD和LFS两模块的模型文件
- mixblock部分尚未开发，model.py的用法是直接将两个模块进行concat随后进入全连接层
- 关于其他的模型文件.py是随着F3net的结构加以改进，与f3net本身没有很大的关系

建议不用管我们的train文件，需要使用的话请按照原论文的参数设置自己写一个train模型的脚本文件
我们的my_train.py仅供参考