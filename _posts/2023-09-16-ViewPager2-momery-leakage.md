---
layout: post
title: ViewPage2 使用过程中产生的内存泄漏
tag: Android
---
# ViewPager2

Google 推出的 Android 平台的组件，意图替代 ViewPage。在使用过程中会比较容易产生内存泄漏。

# 常见的泄露情况

1. 在 fragment 中使用；
2. 搭配 Navigation 使用；

# 解决方案

1. 在 fragment#OnDestroyView 方法中将 ViewPager2 的 adapter 设置为空；
2. 使用 FragmentStateAdapter 时，将 FragmentManager 传入 childFragmentManager，并且将 lifecycle 指定为其 viewLifecycleOwner.lifecycle；
3. 如果你使用的时kotlin，可以在 View#doOnDetach 时也将 ViewPager2 的 adapter 设置为空（Java 可以直接参考此扩展函数内部实现，给 view 设置一个监听，当期 detach 时，进行 adapter 置空）；

# 后续

在调查过程中，看到 google 的 issuetracker 中，开发者承认目前 ViewPager2 存在内存泄漏的问题，但是由于人员短缺，暂时没有进行维护，看 release note 也可以看出，自从 1.0 正式版推出之后，直到 2023 年 1.1 版本也没有发版，仍在测试当中。不过开发者承诺 ViewPager2 仍在维护当中，并且后续版本将会修复这些已知问题。
