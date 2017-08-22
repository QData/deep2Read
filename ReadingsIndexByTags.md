---
layout: page
title: Readings ByTag
desc: "Deep Learning Readings Organized by Tags"
---


Click on a tag to see relevant list of readings.

<ul class="tags">
{% assign sorted = site.tags | sort %}
{% for tag in sorted %}
  {% assign t = tag | first %}
  <li><a href="{{ site.baseurl }}/tag/#{{t | downcase | replace:" ","-" }}">{{ t | downcase }}</a></li>
{% endfor %}
</ul>

---

{% assign sorted = site.tags | sort %}
{% for tag in sorted %}
  {% assign t = tag | first %}
  {% assign posts = tag | last %}

<h4><a name="{{t | downcase | replace:" ","-" }}"></a><a class="internal" href="{{ site.baseurl }}/tag/#{{t | downcase | replace:" ","-" }}">{{ t | downcase }}</a></h4>
<ul>
{% for post in posts %}
  {% if post.tags contains t %}
  <li>
    <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>
    <span class="date">- {{ post.date | date: "%B %-d, %Y"  }}</span>
  </li>
  {% endif %}
{% endfor %}
</ul>

---

{% endfor %}
