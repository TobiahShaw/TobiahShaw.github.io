---
layout: post
title: 快速排序
---
## 一、简介

快速排序（Quicksort）是对冒泡排序的一种改进。快速排序可以说是使用频率最高的排序方法了。它的基本思想是：通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据都比另外一部分的所有数据都要小，然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列。  

快速排序之所以比较快，是因为相比冒泡排序，每次交换是跳跃式的。每次排序的时候设置一个基准点，将小于等于基准点的数全部放到基准点的左边，将大于等于基准点的数全部放到基准点的右边。这样在每次交换的时候就不会像冒泡排序一样只能在相邻的数之间进行交换，交换的距离就大得多了。因此总的比较和交换次数就少了，速度自然就提高了。当然在坏的情况下，仍可能是相邻的两个数进行了交换。因此快速排序的差时间复杂度和冒泡排序是一样的，都是O(N^2)，它的平均时间复杂度为O(NlogN)。其实快速排序是基于一种叫做“二分”的思想。  

## 二、简单实现

```
/**
*简单的快速排序的实现
*/
#include <stdio.h>

int arr_quicksort[101];
void main(){

	int i,n;
	//读入数据
	scanf("%d",&n);
	for(i=1;i<=n;i++){
		scanf("%d",&arr_quicksort[i]);
	}
	quicksort(1,n); //快速排序调用
	//输出排序后的结果
	for(i=1;i<=n;i++) {
		printf("%d ",arr_quicksort[i]);
	}
}

void quicksort(int left,int right){
	int i,j,t,temp;
	if(left>right){
		return;
	}
	temp=arr_quicksort[left]; //temp中存的就是基准数
	i=left;
	j=right;
	while(i!=j){
		//顺序很重要，要先从右往左找
		while(arr_quicksort[j]>=temp && i<j){
			j--;//再从左往右找
		}
		while(arr_quicksort[i]<=temp && i<j){
			i++;
		}
		//交换两个数在数组中的位置
		if(i<j){
			t=arr_quicksort[i];
			arr_quicksort[i]=arr_quicksort[j];
			arr_quicksort[j]=t;
		}
	}
	//终将基准数归位
	arr_quicksort[left]=arr_quicksort[i];
	arr_quicksort[i]=temp;
	quicksort(left,i-1);//继续处理左边的，这里是一个递归的过程
	quicksort(i+1,right);//继续处理右边的，这里是一个递归的过程
}
```

## 三、适用场景  

快速排序的使用场景十分广，基本上是最常用的排序算法了。
