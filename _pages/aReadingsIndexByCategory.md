---
classes: wide
toc: false
author_profile: true
sidebar:
  title: "Reviews Indexed"
  nav: sidebar-sample
layout: single
permalink: /aReadingsIndexByCategory/
title: Readings ByCategory
desc: "Deep Learning Readings Organized by Category Table (2017 to Now)"
order: "9a"
---
<p><a name="topPage"></a></p>


Click on a category to see relevant list of readings.


<ul class="tags">
{% assign sorted = site.categories | sort %}
{% for tag in sorted %}
  {% assign t = tag | first %}
  <li><a href="{{ site.baseurl }}/aReadingsIndexByCategory/#{{t | replace:" ","-" }}"  >{{ t }}</a></li>
{% endfor %}
</ul>

---

{% assign sorted = site.categories | sort %}
{% for tag in sorted %}
  {% assign t = tag | first %}

<BR>
<BR>

<h1><a name="{{t | replace:" ","-" }}"></a><a class="internal" href="{{ site.baseurl }}/aReadingsIndexByCategory/#{{t | replace:" ","-" }}">{{ t  }}</a></h1>

<!--- for each tag, get a table of index -->
<table id="datatab3" summary="Table of readings" border="1">
<tr>
 <h3><b>
  <th>No.</th>
  <th>Date</th>
  <th>Title and Information</th>
  <th>PaperYear</th>
  </b>
  </h3>
</tr>

{% assign counter = 0 %}
{% assign sortedp = site.posts  | sort: 'date' | reverse  %}
{% for post in sortedp %}
  {% if post.categories contains t %}
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

<!--- for each tag, present its posts in orders -->

<div class="posts">

{% assign counter = 0 %}
{% assign sorted = site.posts  | sort: 'date' | reverse  %}
{% for post in sorted %}
  {% if post.categories contains t %}
    {% assign counter=counter | plus:1 %}

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

</div>

---

{% endfor %}

---


<div style="position: fixed; bottom: 39px; right:10px; width: 129px; height: 318px; background-color: #FFCF79;">

{% assign counter = 350 %}
{% assign sorted = site.categories | sort %}
{% for tag in sorted %}
  {% assign t = tag | first %}
  {% assign counter=counter | minus:25 %}
<a href="{{ site.baseurl }}/aReadingsIndexByCategory/#{{t | replace:" ","-" }}"        style="position: fixed; bottom:{{counter}}px; right:10px;">{{ t }}</a>
{% endfor %}


<a style="position: fixed; bottom:40px; right:10px;" href="#topPage" title="Back to Top">BackTop</a>
</div>

