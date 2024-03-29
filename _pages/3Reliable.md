---
toc: true
author_profile: true
sidebar:
  title: "Reviews Indexed"
  nav: sidebar-sample
layout: category
permalink: /categories/3Reliable/
taxonomy: 3Reliable
entries_layout: list
classes: wide
title: 3Reliable
desc: "Recent Readings for Trustworthy Properties of Deep Neural Networks (since 2017)"
order: "3"
---




<p><a name="topPage"></a></p>

  <hr> 
  <h1 class="page-title">{{ page.desc }} (Index of Posts):</h1>

<table id="datatab3" summary="Table of Readings" border="1">
<tr>
 <h3>
  <b>
  <th>No.</th>
  <th>Read Date</th>
  <th>Title and Information</th>
    <th>We Read @</th>
  </b>
  </h3>
</tr>

{% assign counter = 0 %}
{% assign sortedp = site.posts  | sort: 'date' | reverse  %}
{% for post in sortedp %}
  {% if post.categories contains page.title %}
    {% assign counter=counter | plus:1 %}

  <tr>
  <td>{{ counter }}</td>
  <td><span class="date"> {{ post.date | date: "%Y, %-b, %-d "  }}</span></td>
  <td><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></td>
    <td>{{ post.desc }}</td>
  </tr>
  {% endif %}
{% endfor %}

</table>



<!--- present its posts in orders -->


<hr>
<br>
<h1>Here is a detailed list of posts!</h1>
<br>


{% assign counter = 0 %}
{% assign sorted = site.posts  | sort: 'date' | reverse  %}
{% for post in sorted %}
  {% if post.categories contains page.title %}
    {% assign counter=counter | plus:1 %}

<div class="posts">
<hr>
  <div class="post">
    <h2 class="post-title">[{{ counter }}]:
      <a href="{{ site.baseurl }}{{ post.url }}">
        {{ post.title }}
      </a>
    </h2>

    {% if post.date %}
     <span class="post-date">read on: -  {{ post.date | date_to_string }}</span> <br>
    {% endif %}

    {% if post.tags %}
     {% for word in post.tags %}
        {% assign wordd = word | downcase %}        
        <a class="button" href="{{ site.baseurl }}/aReadingsIndexByTags/#{{wordd | replace:" ","-" }}"> {{ word }}</a> 
      {% endfor %}  
    {% endif %}

    {% if post.content contains '<!--excerpt.start-->'  %}
      {{   post.content | split:'<!--excerpt.start-->' | first  }}
    {% else %}
      {{ post.content }}
    {% endif %}

  </div>

  {% endif %}
{% endfor %}

<hr>
<hr>
<br>
<h1>Here is a name list of posts!</h1>
<br>



<div style="position: fixed; bottom: 39px; right:10px; width: 129px; height: 58px; background-color: #FFCF79;">
<a style="position: fixed; bottom:40px; right:10px;" href="#topPage" title="Back to Top">BackTop</a>
</div>
