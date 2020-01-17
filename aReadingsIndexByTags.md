---
layout: page
title: Readings ByTag
desc: "Deep Learning Readings Organized by Detailed Tags (2017 to Now)"
order: "16"
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
  <a class="button"  href="{{ site.baseurl }}/aReadingsIndexByTags/#{{t }}">{{ t }}</a> 
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
  <tr>
  <td>{{ counter }}</td>
  <td><span class="date"> {{ post.date | date: "%Y, %-b, %-d "  }}</span></td>
  <td><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></td>
  <td>{{ post.desc }}</td>
  </tr>
  {% assign counter=counter | plus:1 %}
{% endfor %}
</table>


<!--- for each tag, present its posts in orders -->

<div class="posts">

{% assign sortedp = posts  | sort: 'date' %}
{% for post in sortedp %}

<hr>
  <div class="post">
    <h2 class="post-title">
      <a href="{{ site.baseurl }}{{ post.url }}">
        {{ post.title }}
      </a>
    </h2>

  {% if post.date %}
     <span class="post-date">read on: -  {{ post.date | date_to_string }}</span> 
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

<!--   {{ post.content }}  -->
    <div>Please click above post URL for its content details.</div>

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

