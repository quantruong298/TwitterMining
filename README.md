# Topic Modelling with Twitter Data
This project analyzes how to find hidden topics from social media users data. To answer this question, I used the Twitter API to get the data from the timeline of a specific Twitter user, then applied the LDA (Latent Dirichlet Allocation) model to extract hidden topics. My results show that the LDA algorithm is effective for topic modeling and how to apply it to Twitter. However, the performance may not be stable due to the character limit of tweets. From a safety perspective, this study shows the potential to develop a system that can analyze vast amounts of data.

How to run
==================================

Server
--------
Browser to /server folder:

    cd server
    
Run file server.py:

    python server.py

Client
--------
Browser to /client folder:

    cd server
    
Run react project:

    npm install
    npm start
