# 应用安装  

1. PackageInstallerActivity -> PackageInstallerSession#commitLocked -> PMS

2. PMS#installStage正式安装：
   1. copy安装包
      1. PMS#installStage：
         1. 获取INIT_COPY类型的Message
         2. ***创建install params***
         3. 发送msg
      2. PMS.PackageHandler.handleMessage处理msg：
         1. 取出HandlerParams（InstallParams）对象
         2. ***调用connectToService***
      3. PMS.PackageHandler.connectToService:
         1. 绑定DefaultContainerService，成功后发送Message，仍然由PMS.PackageHandler处理
      4. PMS.PackageHandler.handleMessage处理msg：
         1. ***从等待安装的队列mPendingInstalls中取出第0个（HandlerParams），调用起startCopy方法，最终调用InstallParams#handleStartCopy***
      5. ***InstallParams#handleStartCopy:***
         1. 设置安装标识位（手机内部、sd卡）
         2. 判断apk安装位置
         3. 如果安装位置合法创建一个InstallArgs（FileInstallArgs），调用其copyApk方法
      6. FileInstallArgs#copyApk
         1. 调用doCopyApk
      7. ***FileInstallArgs#doCopyApk***
         1. 创建安装目录
         2. 调用DefaultContainerService服务copyApk包
         3. Copy 三方so到安装位置
   2. 挂载代码
      1. InstallParams#startCopy:
         1. handleStartCopy调用结束后调用handleReturnCode，最终调用processPendingInstall方法处理安装
      2. processPendingInstall：
         1. 检查安装环境，不正常清除拷贝的文件
         2. installPackageTraceLi/installPackageLi
         3. 处理安装完后的操作
      3. ***installPackageLi：***
          1. PackageParser#parserPackage解析apk，主要解析manifest文件
          2. 校验签名信息，比较项目清单摘要
          3. dex优化，dex2oat
          4. installNewPackageLi
      4. installNewPackageLi：
          1. scanPackageLi扫描解析apk，保存相关信息到PMS，新建data目录
          2. 安装成功后updateSettingsLi更新设置，如权限
          3. 如果安装失败deletePackageLi
      5. 最后：如果安装成功则发送广播，launcher获取后新建icon
