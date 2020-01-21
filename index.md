---
layout: default
title: Home
Order: "0" 
---

<h1> Website "deep2Read" for UVA Qdata Group's Deep Learning Journal Club </h1>
<br>
<div>
This website includes a (growing) list of papers and lectures we read about deep learning and related.
Feel free to submit <a href="https://github.com/QData/deep2Read" target="_blank" >pull requests</a>
when you find my typos or have comments. The covered materials are by no means an exhaustive list, but are papers that we have read or plan to learn in our reading group.
</div>


<br>

--- 

## About this website:

+ As a group, we need to improve our knowledge of the fast-growing field of deep learning
+ To educate students in our graduate programs, to help new members in my team with basic tutorials, and to improve current members understand advanced topics better, this website includes a (growing) list of tutorials and papers we survey for such a purpose.
+ We hope this website is helpful to people who share similar research interests or are interested with learning advanced topics about deep learning.
+ Please feel free to email me (yanjun@virginia.edu), if you have related comments, questions or recommendations.

---

## About the Sidebar and how we group the readings 

+ Until Dec. 2019, we have shared reviewes slides of about 300 deep learning papers via this web site. 

+ Until Dec. 2019, we have finished  {{ site.posts | size }} reading sessions. Each reading session includes multiple deep learning papers we reviewed. Their survey slides we made is summarized as blog posts in this website. Each post summarizes one of our reading session. 

+ Due to the large number of slides, we try to organize them according to three different variables. They each maps to one item in this website's side bar. 

+ ByCategory: we group our readings to 10 general groups:  [0Basics]({{ site.baseurl }}/0Basics/) , [1Theoretical]({{ site.baseurl }}/1Theoretical/) , [2Architecture]({{ site.baseurl }}/2Architecture/) , [2Graphs]({{ site.baseurl }}/2Graphs/) , [3Reliable]({{ site.baseurl }}/3Reliable/) , [4Optimization]({{ site.baseurl }}/4Optimization/) , [5Generative]({{ site.baseurl }}/5Generative/) , [6Reinforcement]({{ site.baseurl }}/6Reinforcement/) , [7MetaDomain]({{ site.baseurl }}/7MetaDomain/) , [8Scalable]({{ site.baseurl }}/8Scalable/) , [9DiscreteApp]({{ site.baseurl }}/9DiscreteApp/)  . Please check out these categories and their relevant readings via [Readings ByCategory]({{ site.baseurl }}/aReadingsIndexByCategory/) . We also add each category's page URL as a single item (sorted by Category Names) in our side bar. 


+ ByTags: we use 150 different tags to organize deep learning papers we reviewed. Please check out these tags via [Readings ByTag]({{ site.baseurl }}/aReadingsIndexByTags/) . 

+ ByReadDates: our review slides were done in multiple different semesters (2017Fall, 2018Club, 2019Spring, 2019Fall). To help students navigate, we also group them according to when the slides were  delivered as a talk in our seminar courses or reading club. Please check out our readings sorted by the semester they were made via [Readings ByReadDate]({{ site.baseurl }}/aReadingsIndexByDate/) 


<hr>


## Claim 
+ The covered tutorials and papers are by no means an exhaustive list, but are topics which we have learned or plan to learn in our seminar courses and reading groups.


<hr>



## History

+ This website was started from two seminar courses I taught at UVA  in Fall 2017 and Spring 2019. Later I expand the content with my team reaing group
+ The seminar courses offer opportunities for students to have in-depth understanding and hands-on experience of deep learning. Students are expected to generate top-tier publications when finishing the course.
+ The materials aim to offer opportunities for students to have in-depth understanding and hands-on experience of advances in deep learning. 



### Course Basics and General Description 

> (The following content is for when we have the seminar course)

+ This is an advanced graduate-level deep learning course.
+ The course takes the form of half-seminar and half-project. The form of seminar focuses on paper readings.
+ This course offers opportunities for students to get into research topics about the
 state-of-the-art advanced deep learning.
 + No text book
 + Sit-in: No.  This course is for registered students only.


### Instructor

+ [Prof. Yanjun Qi](http://www.cs.virginia.edu/yanjun/)
+ EMail: [yanjun@virginia.edu](mailto:yanjun@virginia.edu)
+ Rice Hall 503 , 243-3089
+ Office hours: Wed 9:30am-12:30pm.


### Prerequisite:
+ Instructor's Permission for enrollment is required for this course.
+ Required courses as prerequisite: Graduate-level machine
 learning; Introduction of Deep Learning and Graduate-level Optimization are preferred.
+ Familiar reading of [Basic Deep Learning]({{ site.baseurl }}{% link Basic2LearnDeep.md %})  are preferred.


### Course Grading Policy
  The grade will be calculated as follows:
+ 60% for the in-class paper presentations/discussions/ note taking
+ 40% for the project 


### Assignments
+ Sharelatex/overleaf to submit lectures about the assigned papers
+ Each class, we will assign 4 to 6 reading materials (video lectures or papers or research lecture slides )

+ Each student is expected to have three sets of assigntments: 
    
+ (a) Weekly project summary should be updated per week right before project meetings with Prof. Qi; 
+ (b) Assigned presentation slides: please use the BEAMER template shared through the course overleaf project. Please make sure the presentation slides are ready before every Friday 8am;  ([One Example Slide Presenation]({{site.baseurl}}/talks/20171130-Ji.pdf))
+ (c) Assigned scribe notes: please use the latex template shared through the course overleaf project. Please make sure the scribe notes are ready one week after. ([One Example Scribe note](http://www.cs.princeton.edu/courses/archive/spring10/cos424/slides/3-notes-valentino-misener.pdf))

+ For both the paper presentations and the scribe notes, please use the following structure as reference: 

0. Full reference of the paper
1. Motivations / Why needed ? / Why important ?
2. Previous solutions
3. Key insights
4. Key equations
5. Key conclusions
6. Goals achieved: / Under what restrictions or assumptions;



### Logistics Information
+ Announcements are being emailed to the course mailing list.
+ A welcome note will be sent to the mailing  list early in the semester.
+ Errata and answers to questions are being discussed and answered
   on the course emailist.


<hr>

### The Course  Schedule Reference : [The official Academic Calendar at UVA Registrar](http://www.virginia.edu/registrar/calendar.html)




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