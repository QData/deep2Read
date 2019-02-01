---
layout: page
title: Readings ByReadDate
desc: "Our Reviews of Deep Learning Readings by Date-Read"
order: 12
---

<p><a name="topPage"></a></p>

<div class="posts">

{% assign sorted = site.posts  | sort: 'date' | reverse %}
{% for post in sorted %}

<!---<hr> -->
  <div class="post">
    <!---<h1 class="post-title"> 
      <a href="{{ site.baseurl }}{{ post.url }}">
        {{ post.title }}
      </a>
    </h1> 

    <span class="post-date">- {{ post.date | date_to_string }} </span> -->

    <a href="{{ site.baseurl }}{{ post.url }}">
        {{ post.title }}
      </a> - {{ post.date | date_to_string }}

    {{ post.content }}
  </div>
{% endfor %}

</div>



<div style="position: fixed; bottom: 76px; right:10px; width: 88px; height: 36px; background-color: #FFCF79;">
<a style="position: fixed; bottom:80px; right:10px;" href="#topPage" title="Back to Top">BackTop</a>
</div>


<hr>
