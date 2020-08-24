# Subtheme Sentiment Analysis Task:
## Description:
Take the following review for example:

> **“One tyre went missing, so there was a delay to get the two tyres fitted. The way garage dealt with it was fantastic”**

In this review there are numerous insights, insights we call “subtheme sentiments”. A Subtheme sentiment is generally a sentiment towards an aspect or a problem. If we look at the subtheme sentiments of the above review we will get a clearer sense what these generally are.

> **[incorrect tyres sent negative]**      **[garage service positive]**       **[wait time negative]**
                
The main difference between these subthemes is that garage service and wait time are aspects of the service that can be positive or negative while
Incorrect Tyres sent denotes a problem, something inherently negative.
### Approach:
1) Multi-class classification of sentiment.
2) Attaching labels (*aspects*) based on regex rules.
