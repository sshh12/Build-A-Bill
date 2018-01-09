# Build A Bill

Creating Legislation with AI

Inspired by [Andrej Karpathy's Post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## Generated Gibberish Examples

![example 1](https://user-images.githubusercontent.com/6625384/34746534-f2d8ef7a-f559-11e7-8de5-eb6048f2d3f0.png)
![example 2](https://user-images.githubusercontent.com/6625384/34746599-304dc2b8-f55a-11e7-8b4b-fc6b62e43dd2.png)
![example 3](https://user-images.githubusercontent.com/6625384/34746623-4d71f24c-f55a-11e7-819e-52b22f769c1f.png)

## Lab

##### CollectData

This downloads HTML copies of bills using the [GovTrack](https://www.govtrack.us) API.

##### ProcessAndTrainBillModel

This converts bills into trainable data and then trains an LSTM to predict/generate its own bill at a character level.

##### BuildBills

This experiments with different text generation settings.

##### tests/

Folder contains several generated bills.