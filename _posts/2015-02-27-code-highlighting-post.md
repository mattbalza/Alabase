---
layout: post
title: A/B testng and conversion optimization done right (part 1)
excerpt: "Demo post displaying the various ways of highlighting code in Markdown."
tags: [ab testing, a/b testing, conversion optimization, web marketing]
modified: 2015-02-27
author: ECO
comments: true
---

{% include katex_import.html %}

In this series, I will assume that you know what [A/B testing](http://en.wikipedia.org/wiki/A/B_testing) is and that you are interested in giving it a try, but you are not quite sure how to interpret the results.

### Common Pitfalls

Often times, people run A/B tests and then:

* as the tests are running, as soon as there is some prevalence of one result the test gets stopped
* keep running some basic, frequentist[^1] test and as soon as the p-value drops below 5% stop the test
* come up with some other, creative way to stop the test at the wrong time and draw unjustifiable conclusions

[^1]: <http://en.wikipedia.org/wiki/Frequentist_inference>

You might have already read articles about the perils of [peeking while running A/B tests](http://www.evanmiller.org/how-not-to-run-an-ab-test.html), but instead of indulging into difficult statistical issues, here we seek to give you practical advice for how to **get your A/B tests right**.

In this first post, we deal with the two-variant case and depict two statistically sound ways to go. The approach extends naturally to testing multiple variants at once, but we postpone that discussion to a later post.

### The Stats
---
*If you know nothing about probability and statistics, feel free to skip ahead. Consider, however, the option of learning something more about this topic before you get involved in any important project where probability and stats are involved.*

---

Suppose that your variants result in different conversion rates <span class="equation" data-expr="\displaystyle \theta_A, \theta_B" />, and that neither variant has a 0% or 100% conversion rate. A basic A/B test may seek to find whether <span class="equation" data-expr="\displaystyle \theta_A > \theta_B"></span>.

We take a Bayesian approach, and use [Beta(1,1)](http://en.wikipedia.org/wiki/Beta_distribution) priors for each conversion rate. This means that, before seeing any observation, we are quite ignorant as to the distribution of the conversion rates, and think that each rate may well take any value between 0% and 100%. For the observations, we have naturally
{% raw %}
<div align="center" class="equation" data-expr="\displaystyle p(x_i) = {N_i \choose x_i} \theta_i^{x_i} (1 - \theta_i)^{N_i - x_i}" style="margin-bottom: 20px"></div>
{% endraw %}

which is just a [Binomial distribution](http://en.wikipedia.org/wiki/Binomial_distribution).
Combining our prior beliefs with the observations results in the following posterior distribution for <span class="equation" data-expr="\displaystyle \theta_i" />:
{% raw %}
<div align="center" class="equation" data-expr="\displaystyle p(\theta_i | x_i) = \text{Beta}(x_i + 1, N_i - x_i + 1)" style="margin-bottom: 20px"></div>
{% endraw %}
All this says is that what we learn about one conversion rate after a number of visits has a very simple dependency on the number of visitors that we converted and the number of those that we did not convert. Duh...

{% figure anim_beta gif 'Evolution of the posterior distribution of the conversion rate. By the end of the animation, there are 1000 observations (visitors). Notice how the uncertainty is reduced as more observations are accumulated.' %}

Now we have a way of quantifying what we learn about each conversion rate from an experiment, but we still need a way to perform inferences about what conversion rate is higher, and we would like to know by how much one rate is better and have a measure of confidence for our statements. For instance, we might want to say that we are 95% confident that <span class="equation" data-expr="\displaystyle \theta_A - \theta_B \ge 0.05"></span>, i.e. that the conversion rate of variant 1 is at least 5% greater than that of variant 2.

While formulas can be found to answer special questions of this form, the general case requires that we resort to some form of approximation. Here, we choose to use a [Monte Carlo](http://en.wikipedia.org/wiki/Monte_Carlo_method) approximation because it is both convenient and extremely accurate (so much so that you may well disregard the fact that it is an approximation). The following Python snippet, lets us answer the question: with `alpha` confidence, is variant A's conversion rate better than variant B's conversion rate by at least `gamma`? Or is variant B's conversion rate better than variant A's by `gamma`? 

#### Approach #1

{% highlight python %}
import numpy as np

def ab_test(alpha, gamma, visitors_A, conversions_A,
  visitors_B, conversions_B, n_points=1000000):
  y = np.random.beta(conversions_A + 1,visitors_A - conversions_A + 1, n_points) - \
    np.random.beta(conversions_B + 1, visitors_B - conversions_B + 1, n_points)
  # Probability that theta_A > theta_B by gamma
  a_better = np.sum(y >= gamma) / float(y.shape[0])
  # Probability that theta_B > theta_A by gamma
  b_better = np.sum(y <= -gamma) / float(y.shape[0])
  # 95% credible interval
  interval = (np.percentile(y, 5), np.percentile(y, 95))
  print interval
  if a_better > alpha:
    return ['A', a_better, b_better, interval]
  else:
    if b_better > alpha:
      return ['B', a_better, b_better, interval]
    else:
      return ['Inconclusive', a_better, b_better, interval]
{% endhighlight %}

Now we may use the code like this
{% highlight python %}
# 1000 visitors and 10 conversions for A, 2000 visitors and 50 conversions for B
# Is the difference between the rates greater than 0.5% in either direction with
# 95% probability?
ab_test(0.95, 0.005, 1000, 10, 2000, 50)
{% endhighlight %}

This approach has a few advantages over the classical approaches that the internet is full of. Most importantly:

* Feel free to peek. As soon as the test gives a conclusive result, you can stop the test and be confident.
* When you report your result, if you include the credible interval there is a 95% probability that the difference between the two conversion rates lies in there.

**This is no silver bullet, however.** While it is true that this test enjoys very desirable properties from a statistical perspective, beware that it might take a very large amount of visitors before the test terminates, depending on the difference in conversion rates, on the confidence you require (`alpha`) and on the difference you seek (`gamma`). In fact, for values of `gamma` greater than 0, this test might not terminate at all - like...ever!

#### Approach #2

The problem with approach #1 was that it might have never terminated depending on the difference we sought to be confident about. What if, instead of a given difference, we seek a given accuracy in our measure of the difference? Then our test takes three outcomes: one for each variant being better, and one for the two variants being practicly equivalent[^2]. We proceed as follows:

* Choose a region of practical equivalence (ROPE), which is a region within which we consider the two conversion rates to be identical. Outside of this region, on one side we consider variant A being better, on the other side we consider variant B to have the higher conversion rate. 

{% figure rope_example png 'Example of the decision regions given a ROPE going from -0.1 to 0.1' %}

* Choose a desired accuracy in the measurement of the difference of conversion rates. This is done by choosing a credible inteval's amount of probability and width. For example, we might want to say that 99% of the probability should lie in a 0.01-wide range. Think of this as the resolution you want to have for the measurement before terminating the test.

* Once we reach the desired accuracy, we stop the experiment. If the entire credible interval around the mean is outside the region of practical equivalence, we call the test conclusive and declare the better option the one in whose region the credible interval lies. Otherwise, we call the two variants equivalent.


[^2]: This idea is hardly new. For some more, readable material on the topic, consult <http://doingbayesiandataanalysis.blogspot.it/2013/11/optional-stopping-in-data-collection-p.html>

### Standard Code Block

    {% raw %}
    <nav class="pagination" role="navigation">
        {% if page.previous %}
            <a href="{{ site.url }}{{ page.previous.url }}" class="btn" title="{{ page.previous.title }}">Previous article</a>
        {% endif %}
        {% if page.next %}
            <a href="{{ site.url }}{{ page.next.url }}" class="btn" title="{{ page.next.title }}">Next article</a>
        {% endif %}
    </nav><!-- /.pagination -->
    {% endraw %}


### Fenced Code Blocks

To modify styling and highlight colors edit `/_sass/_coderay.scss`. Line numbers and a few other things can be modified in `_config.yml`. Consult [Jekyll's documentation](http://jekyllrb.com/docs/configuration/) for more information.

~~~ css
#container {
    float: left;
    margin: 0 -240px 0 0;
    width: 100%;
}
~~~

~~~ html
{% raw %}<nav class="pagination" role="navigation">
    {% if page.previous %}
        <a href="{{ site.url }}{{ page.previous.url }}" class="btn" title="{{ page.previous.title }}">Previous article</a>
    {% endif %}
    {% if page.next %}
        <a href="{{ site.url }}{{ page.next.url }}" class="btn" title="{{ page.next.title }}">Next article</a>
    {% endif %}
</nav><!-- /.pagination -->{% endraw %}
~~~

~~~ ruby
module Jekyll
  class TagIndex < Page
    def initialize(site, base, dir, tag)
      @site = site
      @base = base
      @dir = dir
      @name = 'index.html'
      self.process(@name)
      self.read_yaml(File.join(base, '_layouts'), 'tag_index.html')
      self.data['tag'] = tag
      tag_title_prefix = site.config['tag_title_prefix'] || 'Tagged: '
      tag_title_suffix = site.config['tag_title_suffix'] || '&#8211;'
      self.data['title'] = "#{tag_title_prefix}#{tag}"
      self.data['description'] = "An archive of posts tagged #{tag}."
    end
  end
end
~~~

### GitHub Gist Embed

An example of a Gist embed below.

{% gist mmistakes/6589546 %}

{% include katex_render.html %}