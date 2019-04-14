# Build A Bill

Creating Legislation with AI

Inspired by [Andrej Karpathy's Post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

Update: Added GPT-2 version.

## Generated Gibberish Examples (GPT-2)

Using one of my other projects as a prompt...
```
========================================================================
A BILL



    To reward Shrivu Shankar for creating the UT Finder mobile app.

    Be it enacted by the Senate and House of Representatives of the
United States of America in Congress assembled,

SECTION 1.   SHORT TITLE.

    This Act may be cited as the ``Shrivu Shankar: Make Money, Serve the
People'em'', or the ``SHAVEDSHARI BUDAPREES Appreciation Act''.

SEC. 2. FINDINGS.

    (a) Findings.--Congress finds the following:
           (1) Shrivu said in his 1999 book ``The Genius of Shrivu
        Shankar'' that, as a young innovator, he knew that there were
        few devices with which modern technology could duplicate the
        capabilities of the telegraph.
           (2) Shrivu wrote: ``The most important advantage of the
        telegraph is that it invented the world we know today. Information
        is transmitted as light, and without the need for a lamp or a
        telephone.''
           (3) According to Shrivu, the telegraph helped to propel the
        development of the world in the direction of modern information
        exchange.
    (b) Achievements.--The Shrivu Shankar Institute of Technology (here
referred to in this Act as the ``SHRIKAR INSTITUTE'') has earned
its place as the world's leading provider of high-speed Internet and
telecommunications services, advanced communications devices, and other
infrastructure through its achievements in telecommunications,
electronics, communications, and energy development.
========================================================================
```

Using one of my [college clubs](https://github.com/txconvergent/) as a prompt...
```
========================================================================
A BILL



     To provide the Texas Convergent startup incubator at UT Austin
                 funding for technical coding projects.

    Be it enacted by the Senate and House of Representatives of the
United States of America in Congress assembled,

SECTION 1. SHORT TITLE.

    This Act may be cited as the ``UT Austin Technical Coding
School Act of 2016''.

SEC. 2. FINDINGS.

    The Congress makes the following findings:
            (1) The Federal Government has failed to develop and use
        the standards and standards that will enable and encourage
        the creation of highly qualified, highly competitive, and
        effective, innovative, and sustainable startup
        opportunities for Texas students.
            (2) The Federal government has failed to develop and
        provide students the training and education that will enable
        students to effectively participate in and succeed in
        activities at UT Austin.
            (3) The Federal Government should use the Federal
        Government funds that are made available through the Texas
        Convergent startup accelerator program to--
                   (A) hire more students in the State and local
                educational agencies, which will help students at and
                near UT Austin study the entrepreneurial and
                entrepreneurial skills necessary for a successful startup
                activity; and
                    (B) ensure, to the maximum extent feasible, that
                students participating in the UT Austin program,
                both in terms of their prior experience in

========================================================================
```

## Generated Gibberish Examples (LSTM)

![example 1](https://user-images.githubusercontent.com/6625384/34746534-f2d8ef7a-f559-11e7-8de5-eb6048f2d3f0.png)
![example 2](https://user-images.githubusercontent.com/6625384/34746599-304dc2b8-f55a-11e7-8b4b-fc6b62e43dd2.png)
![example 3](https://user-images.githubusercontent.com/6625384/34746623-4d71f24c-f55a-11e7-819e-52b22f769c1f.png)

## Lab-GPT2

* Run `python download_data.py` to download and preprocess bills
* Using the code from [nshepperd/gpt-2](https://github.com/nshepperd/gpt-2) run `train.py --dataset bills`
* Use `generate_bill.py` with a trained model and scripts provided from repo above to create custom bills.

## Lab

##### CollectData

This downloads HTML copies of bills using the [GovTrack](https://www.govtrack.us) API.

##### ProcessAndTrainBillModel

This converts bills into trainable data and then trains an LSTM to predict/generate its own bill at a character level.

##### BuildBills

This experiments with different text generation settings.

##### tests/

Folder contains several generated bills.
