---
layout: page
title: Readings ByDate
desc: "Our Reviews of Deep Learning Readings by Date-Read"
---


<div class="posts">

  {% for post in site.posts  %}

  <div class="post">
    <h1 class="post-title">
      <a href="{{ site.baseurl }}{{ post.url }}">
        {{ post.title }}
      </a>
    </h1>

    <span class="post-date">- {{ post.date | date_to_string }}</span>

    {{ post.content }}
  </div>
  {% endfor %}

</div>
