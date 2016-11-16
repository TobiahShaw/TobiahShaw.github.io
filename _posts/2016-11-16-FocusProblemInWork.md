---
layout: post
title: OnKeyListener的onKey()方法问题和焦点问题
---
## OnKeyListener的OnKey()方法问题 ##
&#160;&#160;&#160;&#160;在工作中遇到了一些问题，就是在使用OnKeyListener的onKey()方法时，发现在目标设备（固定的硬件，因为工作和硬件有一定关系，这个也是监听硬件按键）上“return true”后事件依然没有被消费，onKey()在debug模式下，发现被多次执行。但是这种情况在我自己的手机上没有出现。  
&#160;&#160;&#160;&#160;自己设置了flag来标识这个方法是否应该被执行。  
## 焦点问题 ##
&#160;&#160;&#160;&#160;这个实体键盘某个按键会让焦点变到某一个控件，且与软键盘变动不一致。且点击事件一样无法被消费。  
&#160;&#160;&#160;&#160;在使用这个按键的控件的属性内设置：

    android:nextFocusLeft="@id/id_of_itself"
    android:nextFocusRight="@id/id_of_itself"
    android:nextFocusUp="@id/id_of_itself"
    android:nextFocusDown="@id/id_of_itself"
    android:nextFocusForward="@id/id_of_itself"
    
&#160;&#160;&#160;&#160;经测试发现有效。另外EditText控件的android:imeOptions属性实际上只有改变enter键显示图标的作用（实际上也可以通过代码判断这个属性，从而做出不同的操作）。