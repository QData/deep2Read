---
toc: false
classes: wide
author_profile: true
sidebar:
  nav: sidebar-sample
layout: single
permalink: /
---

+ This website aims to educate myself (+my collaborators) to catch up with the fast growing AI literature and Deep learning Tech.
+ This website includes a (growing) list of tech materials I read for the above purpose.
+ I hope this website helps people who share similar interests or want to learn similar topics.
+ Please feel free to email me (yanjun.research@gmail.com), if you have  comments, questions or recommendations.


-- By [Dr. Yanjun Qi] @[https://qiyanjun.github.io/Homepage/](https://qiyanjun.github.io/Homepage/) 

---


## About the Sidebar and how we group the readings 


+ Due to the large number of readings, I try to organize them according to three different variables. Each variable maps to one item in this website's top header bar. 



+ ByCategory: for our readings since 2022 on Generative AI, we group our readings to 4 general groups:  [FM Basics]({{ site.baseurl }}//categories/FMBasic/) , [FM Adapt]({{ site.baseurl }}//categories//FMAdapt//) , [FM Risk]({{ site.baseurl }}//categories//FMRisk//) , [FM Multi]({{ site.baseurl }}//categories//FMMulti//)) 

+ ByCategory: for our readings from 2017 to 2020 on deep learning, we group our readings to 10 general groups:  [0Basics]({{ site.baseurl }}//categories/0Basics/) , [1Theoretical]({{ site.baseurl }}//categories/1Theoretical/) , [2Architecture]({{ site.baseurl }}//categories/2Architecture/) , [2Graphs]({{ site.baseurl }}//categories/2Graphs/) , [3Reliable]({{ site.baseurl }}//categories/3Reliable/) , [4Optimization]({{ site.baseurl }}//categories/4Optimization/) , [5Generative]({{ site.baseurl }}//categories/5Generative/) , [6Reinforcement]({{ site.baseurl }}//categories/6Reinforcement/) , [7MetaDomain]({{ site.baseurl }}//categories/7MetaDomain/) , [8Scalable]({{ site.baseurl }}//categories/8Scalable/) , [9DiscreteApp]({{ site.baseurl }}//categories/9DiscreteApp/). 

+ You can also check out these categories as a whole and their relevant readings via [Readings ByCategory]({{ site.baseurl }}/aReadingsIndexByCategory/) . As shortcuts, we also add each category's page URL as a single item (sorted by Category Names) in our side bar. 


+ ByTags: we use 150 different tags to organize deep learning papers we reviewed. Please check out these tags via [Readings ByTag]({{ site.baseurl }}/aReadingsIndexByTags/) . 


+ ByReadDates: our review slides were done in multiple different semesters (2017Fall, 2018Club, 2019Spring, 2019Fall). To help my students (from a specific semester) navigate, I also group our reading sessions according to what semester a review session has happened at (via my seminar courses or via my journal club). Please check out our readings sorted by semesters via [Readings ByReadDate]({{ site.baseurl }}/aReadingsIndexByDate/) 


<hr>


## Claim 
+ The covered readings are by no means an exhaustive list, but are topics that I learned or plan to learn.


<hr>




<!--<table id="datatab3" summary="Table of readings" border="1">
<tr>
 <h3><b>
  <th>No.</th>
  <th>Date-Read</th>
  <th>Title and Information</th>
<th>PaperYear</th>
  </b>
  </h3>
</tr>

{% assign counter = 1 %}
{% assign sorted = site.posts  | sort: 'date' | reverse %}
{% for post in sorted %}

<tr>
<td>{{ counter }}</td>
<td><span class="date"> {{ post.date | date: "%Y, %-b, %-d "  }}</span></td>
<td><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }} </a></td>
<td>{{ post.desc }}</td>
</tr>

{% assign counter=counter | plus:1 %}
{% endfor %}

</table>  


Click on a tag to see relevant list of readings.

<ul class="tags">
{% assign sorted = site.tags | sort %}
{% for tag in sorted %}
  {% assign t = tag | first %}
  <li><a href="{{ site.baseurl }}/aReadingsIndexByTags/#{{t | replace:" ","-" }}">{{ t }}</a></li>
{% endfor %}
</ul>

---  

-->


<!--

Reading sessions organized by Categories. 

Click on a category to see relevant list of readings.

<ul class="tags">
{% assign sorted = site.categories | sort %}
{% for tag in sorted %}
  {% assign t = tag | first %}
  <li><a href="{{ site.baseurl }}/aReadingsIndexByCategory/#{{t | replace:" ","-" }}">{{ t }}</a></li>
{% endfor %}
</ul>

---

{% assign sorted = site.categories | sort %}
{% for tag in sorted %}
  {% assign t = tag | first %}

<h1><a name="{{t | replace:" ","-" }}"></a><a class="internal" href="{{ site.baseurl }}/#{{t | replace:" ","-" }}">{{ t  }}</a></h1>

-->

<!--- for each tag, get a table of index -->


<!--

<table id="datatab3" summary="Table of readings" border="1">
<tr>
 <h3><b>
  <th>No.</th>
  <th>Title and Information</th>
  <th>We Read @</th>
  </b>
  </h3>
</tr>

{% assign counter = 1 %}
{% assign sortedp = site.posts  | sort: 'date' %}
{% for post in sortedp %}
  {% if post.categories contains t %}

  <tr>
  <td>{{ counter }}</td>
  <td><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></td>
  <td>{{ post.desc }}</td>
  </tr>

  {% assign counter=counter | plus:1 %}
  {% endif %}
{% endfor %}
</table>



{% endfor %}


<hr>

--- 
<br>

We also use the following detailed tags to label each read post we finished. 

<br><br>



<div>
{% assign newtags = "" %}
{% assign sorted = site.tags | sort %}
{% for tag in sorted %}
    {% assign wt = tag | first %}
    {% assign t = wt | downcase %}
    {% assign at = t | replace:" ","-" %}

    {% unless newtags contains at %}
      {% assign newtags = newtags | join:',' | append:',' | append:at | split:',' %}
    {% endunless %}
{% endfor %}

{% assign sortednew = newtags | sort %}
{% for t in sortednew %}
  <a class="button"  href="{{ site.baseurl }}/aReadingsIndexByTags/#{{t }}">{{ t }}</a> 
{% endfor %}
</div>

-->
