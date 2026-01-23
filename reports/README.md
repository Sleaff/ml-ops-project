# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [ ] Create a git repository (M5)
* [ ] Make sure that all team members have write access to the GitHub repository (M5)
* [ ] Create a dedicated environment for you project to keep track of your packages (M2)
* [ ] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [ ] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [ ] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [ ] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [ ] Construct one or multiple docker files for your code (M10)
* [ ] Build the docker files locally and make sure they work as intended (M10)
* [ ] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [ ] Write unit tests related to the data part of your code (M16)
* [ ] Write unit tests related to model construction and or model training (M16)
* [ ] Calculate the code coverage (M16)
* [ ] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

76

### Question 2
> **Enter the study number for each member in the group**
>
> Answer:

s195171 - Kenneth Plum Toft <br>
s242726 - Philip Arthur Blaafjell <br>
s243586 - Joakim Dinh <br>
s242723 - Vebjørn Sæten Skre <br>

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:
Filip
--- question 3 fill here ---

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:
We used uv to manage our dependencies. The list of dependencies was auto-generated using uv export uv export --no-hashes --format requirements.txt --output-file requirements.txt. To get a complete copy of our development environment one would have to run the following commands:
git clone <repo_url>
cd <repo_folder>
uv install --requirements-file requirements.txt

All of our team menmbers used uv and we had no issues with dependencies.

We had the most issues with dependencies towards cloud related access. Such as working in the same GCP project and having the right permissions to buckets. We had issues granting access to DVC remote buckets so we had to make the buckets public for easier access.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:
We started with the cookiecutter template from the course (https://github.com/SkafteNicki/mlops_template) and built our project from there. The main stuff we filled out was `src/project/` with the usual files `model.py`, `train.py`, `data.py`, `dataset.py`, and `api.py` for the core functionality. We made a `configs/` folder with `config.yaml` for Hydra to handle hyperparameters, which was super useful for running different experiments without changing code. We created `dockerfiles/` for both training and the API. For testing we split things into `tests/unittests/` (quick tests that run in CI) and `tests/integration/` (slower tests with the actual model).

We used DVC to version control our data. The processed news dataset lives in `data/processed/` and trained models in `models/`. The main thing we changed from the template was using `uv` for dependencies instead of pip, which worked way better for us. We also added a `scripts/` folder for some bash scripts to run training in the cloud. Finally, we set up a bunch of GitHub Actions workflows in `.github/workflows/` to automatically run tests, linting, and pre-commit checks whenever we pushed code. Everything's organized pretty cleanly with data, models, code, and tests in separate places.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:
We used Ruff for linting and code formatting through pre-commit hooks. Ruff runs automatically before every commit to check our code style and fix common issues like unused imports or incorrect spacing. We configured it in pyproject.toml with a 120-character line length. Our pre-commit setup also checks for trailing whitespace and validates YAML files. We added some type hints to important functions but didn't go crazy with it everywhere. For documentation, we wrote docstrings explaining what our main functions do in model.py, train.py, and api.py.

These things are pretty important when working in a group. Without formatting rules, everyone writes code differently and merge conflicts can become a nightmare. Linting catches silly mistakes before they break anything. Type hints help when you come back to code after a week and forgot what parameters a function needs. In our team of four people, having these tools meant we could work on different parts without constantly asking "what does this function expect?" or "why is this failing?". It made code reviews faster since we could focus on actual logic instead of arguing about spaces vs tabs.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We added a model test, which checks if the model weights change during training. We feed the model a some dummy inputs and labels to see if the weights change. Indicating if our training loop works.

Later we added an inference tester which tested if two sample inputs produced an expected output. However this test was removed as when it was added to github action it could not load the model due to the model being git ignored.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:
Name                      Stmts   Miss  Cover   Missing
-------------------------------------------------------
src/project/__init__.py       0      0   100%
src/project/dataset.py       16      0   100%
src/project/model.py         49     15    69%   48-62, 65-72
-------------------------------------------------------
TOTAL                        65     15    77%

We had a total code coverage of 77%. There were also some warnings which we ignored because they were not relevant to our project. These are warnings such as not using the validation function or changing the amount of workers. Even if our coverage was 100% we would still not trust it to be error free. Code coverage only shows which lines of code are executed during testing, not if the tests are meaningful or if all edge cases are covered. There were also many files which are not covered at all and some tests which were removed for simplicity

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

  Yes, we used branches and pull requests. We followed a feature branch workflow
  with PRs for larger features. Dependabot automated dependency update PRs, and CI ran tests before merging.

  However, several areas could be improved. Some commits went directly to main without PRs (like quick fixes, config
  changes), bypassing code review. This happened when changes felt too small for a PR, but does not follow the
  workflows purpose. Our commit messages were sometimes a bit vague ("fixed the uv.lock file") rather than explaining why
  changes were made. Branch naming was also someimes inconsistent as not all branches followed the type/description convention.

  For future projects, we would enforce stricter branch protection (no direct pushes to main), require more
  descriptive commit messages focusing on intent, and establish clearer PR templates with checklists.
  We could also consider implementing a dev branch between main and features to protect main more, especially since it is suppsoed to be deployable production code.
  Even small changes benefit from the PR process. It creates documentation and allows teammates to stay informed. The work
  of making a PR is minimal compared to the information and trackability it provides.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:
We used DVC for data and model version control. We started out using Google Drive as our remote storage, which was easy to set up initially. Later we switched to Google Cloud Storage buckets to better integrate with the rest of our GCP setup and keep everything in one place.

DVC helped us a lot because we could keep our 40k+ news articles and trained model checkpoints out of git. Instead we just track small `.dvc` files that point to where the actual data lives in the cloud. This meant our git repo stayed small and fast. When someone new joined or when we ran training in the cloud, we could just use `dvc pull` and get the latest data and models without having to manually download anything or ask someone to share files.

The best part was that we could version our data just like code, if we later reprocess the dataset or train a new model, DVC can track those changes. We could also go back to an older version if needed. It also made our CI/CD pipeline simpler since workflows could pull the exact data version needed for testing. Overall it was a bit weird at first, but it made collaboration smoother than passing around files manually or having everyone store large datasets locally.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:
We split our CI into separate workflow files in `.github/workflows/`. The main ones are `tests.yaml` for unit testing, `linting.yaml` for code quality checks. Both trigger on pushes to main and on pull requests, so we catch issues before they get merged. Another important one is and `integration-tests.yaml` for testing with actual trained models and real data, which runs automatically every sunday and manually if needed.

For unit testing (tests.yaml), we test across multiple operating systems - ubuntu, windows, and macos - all with Python 3.13. We use uv's built-in caching (`enable-cache: true`) which speeds things up by not re-downloading dependencies every time. The workflow runs our unit tests with pytest and generates a coverage report showing which parts of our code are actually tested. We intentionally removed DVC data pulling from this unit testing workflow to keep tests fast and isolated, our unit tests then use mocked data instead of real files.

For linting (linting.yaml), we run Ruff to check code formatting and automatically fix small issues. This catches stuff like unused imports or inconsistent spacing before anyone even reviews the code. We also have pre-commit hooks that run similar checks locally, so ideally these linting workflows should just pass.

We originally had more complex CI setup that pulled data from DVC and tested with real models, but we realized that was more of an integration test thing. Now our CI is much faster, unit tests finish in under a minute usually. An example run can be seen at https://github.com/Sleaff/ml-ops-project/actions.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:
Filip
--- question 12 fill here ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:
Filip
--- question 13 fill here ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

We used WandB to track our hyper parameter sweep experiments. Looking at the first image we compare validation accuracy of different sets of hyper parameters over epochs. To get a feel for what hyperparameters was working well we ran 9 different sets of configurations and ran the training for 10 epochs to see their performance. We used WanB’s built-in gaussian search method to select hyper parameters within specified ranges. Looking at the first image we compare validation accuracy over number of epochs. This plot would suggest the topmost line «spring-sweep-6» to be most promising, however, as we varied the batch size as a hyp-parameter, image number 2 showes another story. In the topmost figure in image 2 we plotted validation accuracies over optimizer steps rather than epochs, where we can see that the run «denim-sweep-4» actually learned much faster, but had less optimizer steps than «spring-sweep-6». Thus we created another run with the exact same hyp-param configuration as «denim-sweep-4», and ran it for more epoch. The result of which can be viewed as the green line bottom figure of image 2 «graceful-cherry-29», clearly showing superior performance compared to the other configurations. Image three shows how the different parameters correlate to the validation accuracy. A trend is that too large weight decay and too low learning rate is bad for model performance.

Our best run was «graceful-cherry-29» which achieved a validation accuracy of around 96% after around 30 epochs, or 15k steps, of fine tuning. From other plots comparing validation and training loss, there was no sign of overfitting and sustained training would most likely make the model performance continue the upward trend.

Image 1:
<img width="1420" height="817" alt="!!!!!!!!!!!" src="https://github.com/user-attachments/assets/1fdd8372-06e2-41e1-b87a-7e7d2f6239a7" />

Image 2:

<img width="747" height="791" alt="!!!!!!!" src="https://github.com/user-attachments/assets/e56ef2b5-4288-49b3-80df-a41efd39ee86" />


Image 3:

<img width="1418" height="708" alt="Screenshot 2026-01-22 at 17 57 38" src="https://github.com/user-attachments/assets/425b0c47-fdcd-4636-9e4b-cd444c56715b" />

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:
Filip
--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Our debugging process focused on ensuring integrity of the training loop and data. We implemented a «test_model()» function to verify that model parameters changed during training by comparing the initial state of the weight with the state after a short run. We also included a «test_news_dataset()» function that verified our dataset was of correct type. For errors during code development we used VS codes debugging feature. To maintain structure and an organized track record we made use of pythons logging library and saved prints in these log files with dates and times. 

When running training in the hpc each epoch took around 1-1,5 minutes. To inspect if we could increase the efficiency of our code we made use of code profiling. We used hydra to switch between simple, advanced and PyTorch profiling tools, allowing us to create detailed reports of the code performance, from which we saw that there was a huge CPU overhead. To mitigate this bottleneck we made adjustments to our data loaders, by setting num_workers=4, pin_memory=True and persisten_workers=True. This reduced the CPU overhead and made our training more than 3.3 times faster, with each epoch taking around 18 seconds. 

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We managed to write an API for our model using FastAPI. We are using a language model thus we need to input some text. Thus the requester needs to add a title and a text which they want to figure out if its fake or real news. We did this by adding a post method to the API which then feeds the text to a preloaded model which runs through the lifespan of the API. We also added a get method to check the model parameters to see if it they were loaded correctly. To make the API we had to make a prediction function which was simply since we had already implemented pytorch lightning.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

First we tried to get our API to run on a docker container. However we were having issues with loading our pretrained model into the docker run API.

--- question 24 fill here ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We tried load testing while running the API on locally. For this we used locust. We wrote a locust performance test script to feed our API with random datapoints in our dataset to similate each request to the API. We tested with 10 users at the same time sending requests, getting around 0.7 requests per second. The API handeled it very well, but after adding too many worker eventually the API crashed. We found that the response time increased dramatically when first adding users while it started to lower and stabilize after a while.
![load_testing](figures/load_tests.png)


### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:
Vebjørn
--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:
>

Student s243586 was in charge of implementing pytorch lightning to simplify the training pipeline, developing the API and load testing it, creating docker containers for training and API.

