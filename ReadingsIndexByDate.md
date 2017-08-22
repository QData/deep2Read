---
layout: page
title: Readings ByDate
desc: "Deep Learning Readings Indexed by Date-Read"
---

<table id="datatab3" summary="Table of readings" border="1">
<tr>
 <h3><b>
  <th>No.</th>
  <th>Date</th>
  <th>Title and Information</th>
  </b>
  </h3>
</tr>

{% assign counter = 1 %}
{% assign sorted = site.posts  | sort: 'date' %}
{% for post in sorted %}

<tr>
<td>{{ counter }}</td>
<td><span class="date"> {{ post.date | date: "%Y, %-b, %-d "  }}</span></td>
<td><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }} </a></td>
</tr>

{% assign counter=counter | plus:1 %}
{% endfor %}

</table>


<hr>

<div class="center">
<a href="{{ site.baseurl }}/ReadingsIndexByTags/" title="View Readings by Tag">View Readings organized by Tags</a>
</div>
