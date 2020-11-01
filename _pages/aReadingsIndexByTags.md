---
classes: wide
toc: false
author_profile: true
sidebar:
  title: "Reviews Indexed"
  nav: sidebar-sample
layout: single
permalink: /aReadingsIndexByTags
title: Readings ByTag
desc: "Deep Learning Readings Organized by Detailed Tags (2017 to Now)"
order: "9c"
---

<a name="topPage"></a>

---

Besides using high-level categories, we also use the following detailed tags to label each read post we finished. Click on a tag to see relevant list of readings.

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

<a class="button"  style="border:3px; border-style:solid; border-color:#FF0000;"  href="{{ site.baseurl }}/aReadingsIndexByTags/#{{t }}">{{ t }}</a> 
{% endfor %}
</div>

<hr>

<!--- for each tag, get its related in site.tags -->


{% assign tcounter = 0 %}

{% for newtag in sortednew %}
  {% if newtag  != "" %}

  {% assign tcounter=tcounter | plus:1 %}

<a name="{{newtag }}"></a>
<h1>[{{ tcounter }}]: <a class="internal" href="{{ site.baseurl }}/aReadingsIndexByTags/#{{newtag }}">{{ newtag }}</a></h1>

  {% assign sorted = site.tags | sort %}
  {% for tag in sorted %}
    {% assign wt = tag | first  %}
    {% assign t = wt | downcase | replace:" ","-"  %}

    {% if t  == newtag %}
      {% assign posts = tag | last %}

<!--- for each tag, present its posts in a big table -->

### Table of readings

{% assign counter = 0 %}

<div class="posts">

{% assign sortedp = posts  | sort: 'date' | reverse  %}
{% for post in sortedp %}

  {% assign counter=counter | plus:1 %}

  <div class="post">

  <!--  
  {% if post.tags %}
     {% for word in post.tags %}
        {% assign wordd = word | downcase %}        
        <a class="button" href="{{ site.baseurl }}/aReadingsIndexByTags/#{{wordd | replace:" ","-" }}"> {{ word }}</a> 
      {% endfor %}  
    {% endif %}
  -->
    {% if post.date %}
     <span class="post-date">read on: -  {{ post.date | date_to_string }}</span> <br>
    {% endif %}

    {% if post.content contains '<!--excerpt.start-->'  %}
      {{   post.content | split:'<!--excerpt.start-->' | first  }}
    {% else %}
      {{ post.content }}
    {% endif %}

  </div>

{% endfor %}

</div>

---

{% endif %}
{% endfor %}
{% endif %}
{% endfor %}




---


<!-- <div style="position: fixed; bottom: 70px; right:10px; width: 119px; height: 290px; background-color: #FFCF79;">

{% assign counter = 350 %}
{% assign sorted = site.tags | sort %}
{% for tag in sorted %}
  {% assign t = tag | first %}
  {% assign counter=counter | minus:25 %}
<a href="{{ site.baseurl }}/aReadingsIndexByTags/#{{t | replace:" ","-" }}"        style="position: fixed; bottom:{{counter}}px; right:10px;">{{ t }}</a>
{% endfor %} -->

<div style="position: fixed; bottom: 70px; right:10px; width: 119px; height: 30px; background-color: #FFCF79;">
  <a style="position: fixed; bottom:72px; right:10px;" href="#topPage" title="Back to Top">BackTop</a>
</div>

<hr>

