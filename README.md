# RE_U-NET
 This repository is a modified U-NET. Based on Pytorch.</br>
 For learning and operating on small scale dataset.

### Before Start：
This is my first and probably last academic practicum. I hope you enjoy it.
I coded this repository in my early days as a rookie programmer, and it has been with me for a long time. helped me deal with many difficult tasks. Now I'm improving it and making it open-source for all newcomers. I believe it could put a perfect end to my failed academic career.

## Important Issue
* this repository was directly built on latest Google Colab GPU Environment(torch 1.13.1+cu116)</br>
  Personally, I strongly recommend beginners to register a Google account.</br>
  Colab & Google drive can release you from environment setting.
* If you choose to run this repository on Colab.</br>Please read the <font color=orange>re_U_NET_tutorial_Colab.ipynb</font></br>
  This file includes the model structure and basic information about file tree.
* This a tutorial for my research group, and for those who want to start learning basic information about CV and Pytorch.</br></br>
* <font color=red>This repository is NOT a long-term support repository</font></br></br>

* <font color=orange>I did not and will not offer any Architecture Diagram, for reasons below:</font></br>
   1. This model is based on U-NET, which is very common on the Int.
   2. Concluding the Architecture by yourself will improve you rapidly.
  3. ~~Tired of drawing Architecture Diagram~~
## Code Catalogue & Abstract
* <font color=orange>re_NET_lw.py:</font> Including structure of RE_U-NET
* <font color=orange>dataset.py:</font> Generate data loader
* <font color=orange>train.py:</font> model training
* <font color=orange>predict.py:</font> predicting to generate saved image(demo)

## Special Thanks
[Aladdin Persson](https://www.youtube.com/@AladdinPersson)</br>
His [video](https://www.youtube.com/watch?v=IHq1t7NxS8k&t=2s) about how to rebuild a original U-NET inspire me a lot.</br>
Institute: [Urban Mobility Institute of Tongji Uni](https://umi.tongji.edu.cn/index.htm)

## 写给课题组内同门
<font color=red>如果你选了深度学习算法相关课程，切记请不要直接将这份代码作为作业提交！</font></br>
一方面是我不喜欢别人用我的代码逃课，~~另一方面是我已经用这代码逃课很多次了~~</br>
现在几乎所有CV相关老师都见过这份代码。</br></br>
我开源这个项目的主要原因是：</br>
希望能帮助需要学习相关技术或是对此方面感兴趣的非程序出身的学生</br>
希望我自己折腾三个通宵的成果，在未来的某一天，能帮某个天才节省一到两小时的科研时间</br></br>
本人没什么科研能力，只能做点微小的工作，抱歉。



