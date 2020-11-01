---
toc: true
author_profile: true
sidebar:
  title: "Reviews Indexed"
  nav: sidebar-sample
layout: category
permalink: /categories/2GraphsNN/
taxonomy: 2GraphsNN
entries_layout: list
classes: wide
title: 2GraphsNN
desc: "Recent Readings about Graph Deep Neural Networks (we focused on GNN readings in 2019 Spring Seminar Course)"
order: "2c"
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
