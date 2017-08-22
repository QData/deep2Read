---
layout: page
title: Readings ByDate
desc: "Deep Learning Readings Indexed by Date-Read"
---

<ul class="features">

{% assign sorted = site.posts  | sort: 'name' %}
{% for post in sorted %}

    <li><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}   </a><span class="date">- {{ post.date | date: "%B %-d, %Y"  }}</span></li>
{% endfor %}
</ul>
<hr>

<div class="center">
<a href="{{ site.baseurl }}/tag/" title="View Posts by Tag">View Readings organized by Tags</a>
</div>
