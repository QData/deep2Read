---
layout: page
title: Readings ByTag
desc: "Deep Learning Readings Organized by Tags"
order: 11
---
<p><a name="topPage"></a></p>


Click on a tag to see relevant list of readings.

<ul class="tags">
{% assign sorted = site.tags | sort %}
{% for tag in sorted %}
  {% assign t = tag | first %}
  <li><a href="{{ site.baseurl }}/aReadingsIndexByTags/#{{t | replace:" ","-" }}">{{ t }}</a></li>
{% endfor %}
</ul>

---

{% assign sorted = site.tags | sort %}
{% for tag in sorted %}
  {% assign t = tag | first %}
  {% assign posts = tag | last %}

<h1><a name="{{t | replace:" ","-" }}"></a><a class="internal" href="{{ site.baseurl }}/aReadingsIndexByTags/#{{t | replace:" ","-" }}">{{ t  }}</a></h1>

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

{% assign counter = 1 %}
{% assign sortedp = posts  | sort: 'date' %}
{% for post in sortedp %}
  {% if post.tags contains t %}

  <tr>
  <td>{{ counter }}</td>
  <td><span class="date"> {{ post.date | date: "%Y, %-b, %-d "  }}</span></td>
  <td><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></td>
  <td>{{ post.desc }}</td>
  </tr>

  {% assign counter=counter | plus:1 %}
  {% endif %}
{% endfor %}
</table>

<!--- for each tag, present its posts in orders -->

<div class="posts">

{% assign sorted = posts  | sort: 'date' %}
{% for post in sorted %}
  {% if post.tags contains t %}

<hr>
  <div class="post">
    <!---<h1 class="post-title">
      <a href="{{ site.baseurl }}{{ post.url }}">
        {{ post.title }}
      </a>
    </h1>
    <span class="post-date">- {{ post.date | date_to_string }} </span>-->

    <a href="{{ site.baseurl }}{{ post.url }}">
        {{ post.title }}
      </a> - {{ post.date | date_to_string }}
      
    {{ post.content }}
  </div>

  {% endif %}
{% endfor %}

</div>

---
***

{% endfor %}


<div style="position: fixed; bottom: 76px; right:10px; width: 88px; height: 36px; background-color: #FFCF79;">
<a style="position: fixed; bottom:80px; right:10px;" href="#topPage" title="Back to Top">BackTop</a>
</div>

<hr>

