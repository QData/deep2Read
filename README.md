## About
This site is generated by [Jekyll](http://jekyllrb.com) + [Hyde](https://github.com/poole/hyde#readme)  + [PostTagging](https://www.jokecamp.com/blog/listing-jekyll-posts-by-tag/) .

## A list of easy steps to set up a project blog web using Hyde + GitPage

1. set up ruby: please follow the step from [steps](https://gist.github.com/mcls/3118518) for mac version
2. install jekyll
```sh
~ $ gem install jekyll
```
3. download/fork [Hyde](https://github.com/poole/hyde) to your Git, and change to your favorite web name
4. clone it to your local folder : web-name
5. now
```sh
~ $ cd web-name
```
6. now add a file with the name "Gemfile" with the following code inside
```sh
source 'https://rubygems.org'
gem 'github-pages', group: :jekyll_plugins
```
7. now deploy it locally
```sh
~/web-name $ bundle install
~/web-name $ bundle exec jekyll serve
```
8. Now use your favorite browser to open http://localhost:4000
9. now customize the file '_config.yml'. See more tips from [this blog](http://anthony.wiryaman.com/2016/08/17/creating-the-blog/):
- remove 'relative_permalinks: true'
- change the tile: / description: / markdown: to kramdown / highlighter: to rough;
- revise the number of 'paginate:' to control how many posts to show under 'Home' page
- revise 'url:' to the GitPage URL of the Git User, like 'https://user.github.io'  
10. If this blog will be deployed as the user gitPage, change 'baseurl:' in '_config.yml' to '/', then go to step 12.
11. If this blog will be deployed as the gitPage for gitUser/web-name,
- change 'baseurl:' in '_config.yml' to '/web-name/'
- revise 'sidebar.html' under '_includes, line 24, to
```html
            <a class="sidebar-nav-item{% if page.url == node.url %} active{% endif %}" href="{{ site.baseurl }}{{ node.url }}">{{ node.title }}</a>
```
- revise 'index.html', line 11, to
```html
      <a href="{{ site.baseurl }}{{ post.url }}">
```
- revise 'post.html' under '_layouts, line 17, to
```html
          <a href="{{ site.baseurl }}{{ post.url }}">
```

12. add more sidebar items, e.g., "contact" by adding a md file directly, e.g.web-name/contact.md. Add the following two lines into the [front matter](https://jekyllrb.com/docs/frontmatter/) of web-name/contact.md :
'layout: page'
 and 'title: contact'
13. add year-month-date-name.md under '_posts' folder
14. check locally by
```sh
~/web-name $ bundle exec jekyll serve
```
15. push into your git-site, using the site's -> "setting" -> section "GitHub Pages" to setup and publish the web as GitPage

16. add post tagging using codes from [PostTagging](https://www.jokecamp.com/blog/listing-jekyll-posts-by-tag/)

17. add more interesting liquid programming using [Tips](https://gist.github.com/smutnyleszek/9803727)
