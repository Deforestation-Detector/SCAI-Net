System and Unit Test Report

Product Name: Deforestation Detector
Team Name: Deforestation Detector
Team Members:
Donnovan Henry
Sam Edwards-Marsh
John Beresford
Geo Ochoa
Chris Sterza
Date: 3/8/22
System Test Scenarios
Sprint 1
Stories:
As a user, I would like to see an indication of the loading progress of the page, because large media can take many seconds to load and I don’t want to be stuck with a blank screen
As a user, I would like the colors of the website to be fitting to the rest of the experience and aesthetically pleasing, because color can affect readability of text and visibility of other elements.
As a user, I would like the fonts used on the page to be readable and fitting to the purpose of the website.
As a user, I would like to be able to see more about the experience in an about page, because it will allow me to understand its purpose.
Scenario:
1. Visit http://deforestationdetector.com and you should be met with a loading screen
2. The type on the page should be legible and fully visible
3. Click on the Learn More button and you should be met with an about page
Scenario: 
1. Go into the rainforest.ipynb and run all cells
2. User should see all possible neighbors for each image
3. Scenario:
4. Go into the rainforest.ipynb and run all cells
​​5. User can expect to see metrics on model performance for validation
Sprint 2
Stories:
As a user, I want to be able to navigate the UI/experience intuitively, because otherwise I may miss information and/or get frustrated.
As a user, I would like to learn more about the problem in an information page so I can understand its importance.
As a user, I would like the data to be accurate, because misinformation can be more dangerous than ignorance.
As a user, I would feel more comfortable browsing a page with a professional-looking icon.
As a user, I want the data to be relevant to the purpose of the experience
As a user, I would like to arrive at a landing page, because it would make the rest of the experience more accessible
As a user, I would like to be able to visualize the scope of today’s deforestation because it will help me to understand its threat to my future
Scenario:
1. CD into directory with rainforest.py
2. Call rainforest.py and specify a supported base model
3. Train model.
4. User should notice evaluation returns a validation accuracy of over 90%
5. User should notice that one graph appears with seventeen individual confusion matrices, one for each label
6. User should then see a sub-directory for their model under the checkpoints directory, which they can then load.
Scenario:
1. CD into directory with rainforest.py
2. Call rainforest.py and specify a supported base model
3. Initialize Docker container from the Dockerfile. 
4. Train model
5. User should see a great speed increase in training time, and should see notification about graphics card usage.
Scenario:
1. After loading, the user should be met with a landing page
2. The user should click on the explore or the learn more button.
3. User should see an indicator of how to interact with the page if idle for too long on the explore view
4. Mouse interactions with the 3D environment should be intuitive (drag up/down to move the camera back/forward, drag left/right to rotate)
5. The information presented in the learn more view should be direct and not misleading. It should also be verifiable.
6. The user should be able to see an icon in the browser tab.
7. The user should be able to see and interact with the 3D visualization of the Amazon Rainforest

Sprint 3
Stories:
As a user I want to know how much deforestation is due to human intervention rather than natural causes
As a user, I would like to see a proper domain name, because it makes me feel like the information there is more reputable.
As a user, I want to be able to input an image and get a prediction on it.
As a user, I would like calls to action that can lead me to ways of contributing to the efforts against deforestation.
As a user, I want to know which specific regions in the Amazon rainforest are being affected by deforestation.
Scenario:
1. User enters the page and clicks the learn more button
2. The user should be able to find information on what contributes to deforestation and, of those things, what is natural or not.
3. User then exits the learn more view and enters the explore view and clicks on an interactable element.
4. The user should then see more information about that specific element, and calls to action for contribution.
Scenario:
1. CD into directory with rainforest.py
2. Call rainforest.py and specify a supported base model
3. Initialize Docker container from the Dockerfile. 
4. Train model
5. User should see a great speed increase in training time, and should see notification about graphics card usage.
Scenario:
1. CD into directory with rainforest.py
2. Call rainforest.py and specify a supported base model
3. Train model.
4. User should notice evaluation returns a validation accuracy of over 90%
5. User should notice that one graph appears with seventeen individual confusion matrices, one for each label
6. User should then see a sub-directory for their model under the checkpoints directory, which they can then load.

Sprint 4
Stories:

Scenario
1. Navigate to repo with scai_test.py
2. Run `python scai_test.py`
3. user should see all (6) tests pass

Scenario:
1. User enters the website and is displayed an Explore and Learn More button. A mouse hover on any of these changes the button’s brightness.
2. On the Learn More page, a list of sections should appear whose underline property changes on a mouse hover indicating these can be clicked.
3. At the bottom of both the Learn More and Label Pages, a list of websites are displayed whose text color property is changed to orange on a mouse hover.
4. On the Learn More, a section titled “About This Page” displays the satellite images used for the 3D experience.
5. On both the Learn More and Label pages, an “x” appears at the top right which displays an animation indicating we can go back to the page we were at.
6. After clicking the “x”, we are back to the explore page and if we remain idle the cursor shows instructions to use the WASD or arrow keys to move around.
