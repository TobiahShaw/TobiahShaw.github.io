# Android 系统启动流程

1. 上电加载boost loader
2. boost loader拉起操作系统
3. 操作系统启动init进程
4. init进程启动：创建和挂在启动所需文件系统、初始化和启动属性服务、解析init.rc启动Zygote
5. Zygote启动： 创建dalvik/art、从native层进入Java层、创建一个server端的socket等待AMS启动新进程、启动SystemService
6. System Service启动，创建系统服务（AMS，WMS，PMS，Binder线程池）
7. 启动launcher
