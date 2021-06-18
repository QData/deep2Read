---
toc: true
layout: category
permalink: /categories/0Basics/
taxonomy: 0Basics
entries_layout: list
classes: wide
title: 0Basics
desc: "Recent Readings for Basic Topics of Deep Neural Networks (since 2017)"
order: "0"
author_profile: true
sidebar:
  title: "Reviews Indexed"
  nav: sidebar-sample
---


<p><a name="topPage"></a></p>

  <hr> 
  <h1 class="page-title">{{ page.title }} (Index of Posts):</h1>

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

<hr>