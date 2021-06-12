[(3 条消息) Typora 中将多张图片并排/分行显示_M010K 的博客-CSDN 博客_typora 图片并排](https://blog.csdn.net/qq_43444349/article/details/107292803)

[typora-latex-theme/Supplemental at main · Keldos-Li/typora-latex-theme (github.com)](https://github.com/Keldos-Li/typora-latex-theme/tree/main/Supplemental)

### 仿照 Latex

### 第一页

![hust.jpg](https://b3logfile.com/siyuan/1610205759005/assets/hust-20210608145412-8gldwmj.jpg)

```markdown
<div class="cover" style="page-break-after:always;font-family:方正公文仿宋;width:100%;height:100%;border:none;margin: 0 auto;text-align:center;">
    <div style="width:60%;margin: 0 auto;height:0;padding-bottom:10%;">
        </br>
        <img src="https://gitee.com/Keldos-Li/picture/raw/master/img/%E6%A0%A1%E5%90%8D-%E9%BB%91%E8%89%B2.svg" alt="校名" style="width:100%;"/>
    </div>
    </br></br></br></br></br>
    <div style="width:60%;margin: 0 auto;height:0;padding-bottom:40%;">
        <img src="https://gitee.com/Keldos-Li/picture/raw/master/img/%E6%A0%A1%E5%BE%BD-%E9%BB%91%E8%89%B2.svg" alt="校徽" style="width:100%;"/>
	</div>
    </br></br></br></br></br></br></br></br>
    <span style="font-family:华文黑体Bold;text-align:center;font-size:20pt;margin: 10pt auto;line-height:30pt;">《论文名称》</span>
    <p style="text-align:center;font-size:14pt;margin: 0 auto">论文类型 </p>
    </br>
    </br>
    <table style="border:none;text-align:center;width:72%;font-family:仿宋;font-size:14px; margin: 0 auto;">
    <tbody style="font-family:方正公文仿宋;font-size:12pt;">
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">题　　目</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 论文题目</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">上课时间</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 上课时间</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">授课教师</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">教师姓名 </td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">姓　　名</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 你的名字</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">学　　号</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">你的学号 </td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">组　　别</td>
    		<td style="width:%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 你的组别</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">日　　期</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">完成日期</td>     </tr>
    </tbody>          
    </table>
</div>
```

![image.png](https://b3logfile.com/siyuan/1610205759005/assets/image-20210529080021-kosf07o.png)

### 摘要

```markdown
<center><div style='height:2mm;'></div><div style="font-size:14pt;">Author：Jixiong Su</div></center>
<center><span style="font-size:9pt;line-height:9mm"><i>Huazhong University of Science and Technology</i></span>
</center>
<div>
<div style="width:82px;float:left;line-height:16pt"><b>Abstract: </b></div> 
<div style="overflow:hidden;line-height:16pt">In this paper, I use Support Vector Machine to predict donor site in eukaryotic genes, with One-hot coding and different lengths of samples to explore different kernels of SVM. Finally the result is that SVM has the best predictive ability for sequences of 20 upstream sites and 20 downstream sites of exon/intron boundary.
</div>
</div>
<div>
<div style="width:82px;float:left;line-height:16pt"><b>Key Words: </b></div> 
<div style="overflow:hidden;line-height:16pt">SVM, Machine Learning, Splice Site, Gene Finding</div>
</div>
```

![image.png](https://b3logfile.com/siyuan/1610205759005/assets/image-20210608145300-n6prps4.png)

### 图片并排

使用 table

```
　　<table style="border:none;text-align:center;width:auto;margin: 0 auto;">
	<tbody>
		<tr>
			<td style="padding: 6px"><img src="https://gitee.com/Keldos-Li/picture/raw/master/img/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F%E4%BB%BD%E9%A2%9D1.png" ></td><td><img src="https://gitee.com/Keldos-Li/picture/raw/master/img/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F%E4%BB%BD%E9%A2%9D2.png" ></td>
		</tr>
        <tr><td><strong>图 8  全球操作系统市场份额</strong></td><td><strong>图 9  中国操作系统市场份额</strong></td></tr>
	</tbody>
</table>
```

两张图片并排

```html
<center class="half">
    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="300"/>
    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="300"/>
</center>
```

三张图片并排显示的效果如下（width 均为 200）：

```html
<center class="half">
    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="200"/>
    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="200"/>
    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="200"/>
</center>
```

四张图片并排显示的效果如下（width 均为 150）：

```html
<center class="half">
    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="150"/>
    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="150"/>
    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="150"/>
    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="150"/>
</center>
```

若需要将四张图片分两行显示也仅需要调整 width 即可：

```html
<center class="half">
<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="300"/>
<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="300"/>
<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="300"/>
<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggnfolw5kxj30u00u0qe5.jpg" width="300"/>
</center>
```
