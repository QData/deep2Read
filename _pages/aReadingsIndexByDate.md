---
classes: wide
toc: false
author_profile: true
sidebar:
  title: "Reviews Indexed"
  nav: sidebar-sample
layout: single
permalink: /aReadingsIndexByDate/
title: Readings ByReadDate
desc: "Our Reviews of Deep Learning Readings by Date-Read (Now to 2017)"
order: "9b"
---


<p><a name="topPage"></a></p>


Click on a term-tag to see relevant list of readings we finished in a certain semester.


<ul class="terms">
  {% assign terms = site.posts | group_by: 'term' | sort: "name"  | reverse %}
  {% for term in terms %}
    {% if term.name == "" %}
      {% assign t = "NoTerm" %}
    {% else %}
      {% assign t = term.name %}
    {% endif %}

     <li><a href="{{ site.baseurl }}/aReadingsIndexByDate/#{{t | replace:" ","-" }}"  >{{ t }}</a></li>

  {% endfor %}
</ul>


---


{% assign ptcounter = 0 %}


{% assign terms = site.posts | group_by: 'term' | sort: "name"  | reverse %}
  {% for term in terms %}
    {% if term.name == "" %}
      {% assign t = "NoTerm" %}
    {% else %}
      {% assign t = term.name %}
    {% endif %}


<BR>
<BR>


<h1>
  <a name="{{t | replace:" ","-" }}"></a>
  <a class="internal" href="{{ site.baseurl }}/aReadingsIndexByDate/#{{t | replace:" ","-" }}">{{ t  }}</a>
</h1>

---

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
{% assign sortedp = site.posts  | sort: 'date' | reverse %}
{% for post in sortedp %}
  {% if post.term contains t %}

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


{% assign ptcounter=ptcounter | plus:counter %}

<!--- for each term, present its posts in orders -->

<div class="posts">

{% assign counter = 0 %}

{% assign sorted = site.posts  | sort: 'date' | reverse %}
{% for post in sorted %}
  {% if post.term contains t %}

  {% assign counter=counter | plus:1 %}

  <hr>
  <div class="post">
    <h2 class="post-title"> [{{ counter }}]:
      <a href="{{ site.baseurl }}{{ post.url }}">
        {{ post.title }}
      </a>
    </h2>


  {% if post.date %}
     <span class="post-date">read on: -  {{ post.date | date_to_string }}</span> <br>
  {% endif %}

   {% if post.categories %}
      {% for word in post.categories %}
        <a class="button" href="{{ site.baseurl }}/aReadingsIndexByCategory/#{{word | replace:" ","-" }}"> {{ word }}</a> 
      {% endfor %}  
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


{% endfor %}

<hr>

> In total, we have finished number of <b> {{ ptcounter }} </b>  reading sessions.

<hr>

<div style="position: fixed; bottom: 39px; right:10px; width: 119px; height: 318px; background-color: #FFCF79;">

{% assign counter = 350 %}

{% assign terms = site.posts | group_by: 'term' | sort: "name"  | reverse %}
  {% for term in terms %}
    {% if term.name == "" %}
      {% assign t = "NoTerm" %}
    {% else %}
      {% assign t = term.name %}
    {% endif %}

    {% assign counter=counter | minus:25 %}
    <a href="{{ site.baseurl }}/aReadingsIndexByDate/#{{t | replace:" ","-" }}"   style="position: fixed; bottom:{{counter}}px; right:10px;">{{ t }}</a>

{% endfor %}

<a style="position: fixed; bottom:40px; right:10px;" href="#topPage" title="Back to Top">BackTop</a>
</div>

