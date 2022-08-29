---
layout: post
title: Activity Window and View
---

# Activity Window and View

1. View展示
   1. Activity#setContentView方法调用
      1. 内部是调用mWindow的setContentView
      2. mWindow是Activity执行attach时创建的，并将activity作为callback传入，并传递一个WindowManager
      3. mWindow#setContentView检查mContentParent是否为空，空则初始化DecorView和mContentParent，并将传入的布局填充到mContentParent
   2. ActivityThread#handleActivityResume向WMS传递View
      1. onCreat的操作完其实UI还不可见，到resume状态才可见，handleActivityResume调用WindowManager的addView方法
      2. WindowManager实现类是WindowManagerImpl其addView实际是个空壳，调用了WindowManagerGlobal的addView方法
      3. WindowManagerGlobal#addView内部创建了一个ViewRootImpl，并调用其setView方法
      4. ViewRootImpl#setView方法内部调用requestLayout完成View测量和绘制，并mWindowSession的addToDisplay方法将View添加到WMS（WMS#addWindow）
      5. mWindowSession实际是IWindowSession类型，是个Binder类型，真正的实现类是System进程的Session
2. 点击事件
   1. IWindowsSession处理一系列输入管道
   2. InputStage#onProcess调用mView（DecroView）#dispatchPointerEvent方法（实际由View实现）
   3. View#dispatchPinterEventer实际上调用了PhoneWindow的CallBack的dispatchTouchEvent方法
   4. PhoneWindow的CallBack实际是attach时传入的activity，其dispatchTouchEvent调用最终DecorView的dispatchTouchEvent
