1. 写一个test.py脚本，使用viztracer来分析当前的CPU 版本的Mamba模型推理的时间瓶颈在哪里
完成

2. 那么让我们来试一试我的想法，即使用deltaU存储x，然后一次性计算所有x，再一次性计算所有y，首先复制一份model.py为model_new.py，然后在新的文件里面使用我的想法进行实现。然后修改test.py文件，输出一个mamba_new_profile
完成

3. 根据我的实验结果,更新design.md文件，在后面增加这个成功案例，并且补上分析内容
完成

4. 总结我们讨论的c++文件使用方法，前期用JIT后期用setup，把流程整理成文档cpp_workflow.md，记住，一定要简短简洁记录，保留关键信息即可。
完成
5. 完成以下任务：
    - 实现一个selective_scan.cpp，分别使用原始selective scan和我优化过后的selective scan实现selective_scan于selective_scan_fixlen两个函数。
    - 使用cpp_workflow当中的方案1，JIT的方法，修改当前的model_cpp.py文件，实现两种Mamba分别调用上一个任务当中的两个函数
    - 修改test.py脚本，分别测试上一个任务当中的两种Mamba，使用viztracer记录输出结果
  我安装了msvc，帮我修改脚本，改为使用MSVC，JIT的方案来进行测试
  完成

6. 修改当前cpp实现，不要手写矩阵乘法，改成调用torch API的方案。
  完成

7. 修改当前cpp实现，在当前实现的基础上，给Selective_Scan_Cpu和Selective_Scan_Fixlen_Cpu分别在额外实现一个Selective_Scan_Fixlen_Cpu_Pscan和elective_Scan_Cpu_Pscan的方法。
完成

8. 实验证明当前pscan的方法并不可行，那么回退回原来的版本
完成

9. 现在我们使用setup.py的方法，修改test.py于setup.py，我们使用这种方法再次进行测试
完成

10. 修改test.py，对比四种输出结果，确保四种方式输出结果相同
完成

11. 梳理VisionMamba目录，生成vim_analysis.md，分析vision mamba的计算过程，并添加一节比较Mamba和Vision Mamba的计算有什么区别
完成

12. 在VisionMamba_CPU的目录下，编写一个python脚本inf_cpu.py，创建一个Vim tiny模型，随机初始化参数，然后进行一次inference，使用viztracere分析其性能。
完成

13. 现在，我们要使用Mamba_CPU当中的cpp实现来检验优化过的模型结果如何。完成下面的工作
    - 我已经将selective_scan_cpp.cp313-win_amd64.pyd放在ops当中了，现在修改selective_scan_interface.py，创建一个新的函数，使其调用我们使用cpp实现的操作。也就是有3种实现，一种是原始的，一种是cpp的Selective_Scan_Cpu，一种是Selective_Scan_Fixlen_Cpu。
    - 修改mamba的实现，增加一个额外的参数，使其可以选择上面任务当中的Selective_Scan实现
    - 修改inf_cpu.py，初始化一个模型，分别使用三种Selective_Scan方式进行测试，比较它们的输出结果是否相同，使用viztracere分析其性能。
完成

14. VisionMamba_CPU\mamba-1p1p1\mamba_ssm\ops目录下，实现selective_scan.cpp文件，实现cpp版本的selective_scan_ref，包括原始实现和我的fix len实现。并且实现对应的setup.py
  完成

15. 运行setup.py，创建库文件。然后运行inf_cpu.py，检查输出结果
  完成

16. 当前的mamba_simple.py文件存在一些问题，正常来讲应该是V2版本的双向mamba，但是现在CPU版本的只有单向mamba。所以我们需要修改当前实现，那么需要完成以下任务
    - 修改mamba_simple.py中的_forward_reference，传入conv的相关参数
    - 修改mamba_simple.py当中的forward函数，不仅使用图片正向传输，还要反向传播，也就是正向调一次_forward_reference还要反向调一次，然后将结果合到一起，就像是下面v2版本gpu的实现一样
  完成

17. 使用narrow的方式优化当前cpp文件
  完成

18. 修改当前setup.py文件，使其检测当前的系统是win32或者linux，然后将其编译成对应的库文件
  完成

19. 实现一个compare_time.py,创建1个7M参数的CNN和一个7M参数的Vit，输出这两个模型推理用时
完成

20. 修改mamba_simple.py,，现在使用的是self._forward_reference进行一个方向的推理，增加_foward_fuse_reference函数，这个函数将前向和反向输入都收集起来，然后将其拼接成一个，使用一次计算，同时完成正向和反向的计算，就像我们讨论的那样。现在直接用python来实现就可以了。然后再inf_cpu.py中增加这一个实现的测试
完成

21. 梳理整个目录内容，然后重新整理mamba_simple.py当中的_forward_fuse_reference函数，完成以下内容
    - 从步骤5开始，之后的内容使用一个selective_fused_scan_fn函数代替
    - 同样的，可以选择是否使用cpp版本，是否选择fixlen版本
    - 实现一个python版本的selective_fused_scan在selective_scan_interface.py当中
    - 实现一个cpp版本的selective_fused_scan和一个selective_fused_scan_fixlen在selective_scan.cpp当中
完成

22. 修改当前inf_cpu.py文件，现在分别进行python，cpp版本的原始，两阶段，融合实验。
  完成

23. 修改setup.py函数，使其在树莓派下编译能够利用树莓派的加速指令集
  完成

24. 现在使用的模型输入默认为224*224，把它改成128*128的
  128*128不符合我的要求，该会224*224
  完成

25. 优化当前fused方案当中的cat 操作，使用预分配+直接写入的方式进行优化
  完成

26.  优化内存布局，提高访问时缓存命中率
  完成

27. 现在我们做的更激进一点，在mamba-1p1p1当中，实现VisionMamba.cpp，这个VisionMamba.cpp完全实现class Mamba(nn.Module)，bimamba_type 为v2，可以调用C:\workspace\Workspace\Workspace\mamba-minimal\VisionMamba_CPU\mamba-1p1p1\mamba_ssm\ops\selective_scan.cpp中实现的代码，然后实现一个setup.py。最后修改inf_cpu.py，增加一种测试，为全cpp实现，包括原始实现，fixlen，fused和双重优化
  完成

28. 现在，帮我清理当前目录下无关的文件，如C:\workspace\Workspace\Workspace\mamba-minimal\VisionMamba_CPU\RASPBERRY_PI_DEPLOYMENT.md 这样的文件完全没有必要，实验结果文件也可以去掉。把你认为应该清除的文件写成delete.md
    完成

29. 按照delete.md清理文件
    完成
  
30. 编写一个readme.md文件，首先说明这是一个什么项目，然后说明怎么使用，最后分析我们采用了哪些优化方法
  