# Donovan's module and functional testing
I tested the scai_test module. For some functions, unit tests became a little pointless
and meaningless, so I voided those. For instance, any plotting functions just did not
make sense to test. Another caveat worth mentioning: there were other functions that were
just not feasible to test. There were plenty of functions that required a model to test and
thus would take at least 10-20 minutes to test if utilizing the graphics card, and thus
we voided them. One last caveat, there are plenty of functions who will result in exceptions
or run time errors if invalid input is given, but not because our function didn't handle those
cases, rather the libraries required in those setups could not be given invalid input
without throwing exceptions. However, in the vast majority of functions, you'll find that at
the very least, we ensured that the elements passed in were valid in that they're of the right type,
with the right number of elements. This was largely tested in the scai_test module. More details
about each function can be found below:

## set_NLABELS():
### Description and expectations:
#### set_NLABELS is supposed to take in a dataframe. If not, the data is invalid, and the test should fail,
resulting in the NLABELS global variable remaining with the value "None". Furthermore, the dataframe
is meant to contain exactly two columns. First, an image_name
column describing the name of the file within the directory. Second, the corresponding labels for that
image. If valid input is provided to the function, then after the call to it, the NLABELS global
variable will no longer be "None". This is the fact that we leveraged to test input validity.
In terms of testing valid input, we supplied a substantially smaller csv file within a substantially
smaller data file, where we called the set_NLABELS after setting up a dataframe object from that
csv file. After this, we checked the NLABELS value to ensure that the number of labels corresponded
to the number of unique labels within the csv.
### Equivalence classes:
#### Not a dataframe
##### Should and does result in "None"
#### Dataframe with less or more than 2 columns, or with a missing image_names/tags column
##### Should and does result in "None"
#### Dataframe with missing image_names column
##### Should and does result in "None"
#### Dataframe with missing tags column
##### Should and does result in "None"
#### Valid dataframe with 5 unique labels and 6 image_names
##### Shoulda and does result in 5

create_data():
create_data is meant to take in a dataframe object with the file names and the corresponding labels.
We followed the exact same invalid input logic as the function directly above, set_NLABELS. For testing valid
input, we created a dataframe off of the small csv file described above, and passed that as an argument
into the create_data function. After this, the two returned values should be something other than "None".
If so, we know that two data generators were successfully created. The only way that the data generators
could fail to be created is if they threw an exception, or if they returned ("None", "None").

f1():
This is one of the functions with the caveats as mentioned above. This function as passed as an argument into
the tensorflow evaluation metric. This function is bound by a relu/sigmoid activation function, which, in turn,
means that it cannot be outside of the range of 0 or 1. In turn, the only meaningful tests I could think of were
to ensure that the elements provided were 32-bit tensors. If so, then as mentioned, they must be bound between 0
and 1. Furthermore, we hand-tested the function. We provided some arbitrary values between 0 and 1 (for reasons
described several times), and verified that their ouputs were as expected.

f1loss():
This function's testing is literally the exact same as f1 just with different hand-tested, expected outcomes.
This function is just a differentiable version of the f1 function such that it can be used as a loss function
so the model can perform a gradient.

create_transfer_model():
This function is supposed to take in an architecture name specified by a string, initialize it,
and set it as a base model followed by a flattening layer with a dense neural network. With hundreds of millions
of neurons within each base model, testing for model equality here isn't really feasible here, so the way I saw it,
there were two cases between invalid input and valid input. If the input was invalid because the element in the ARCH
variable wasn't a string, or if the specified ARCH isn't in the set of architectures, then "None" is returned. Else
a valid model is returned. This is further backed up by the fact that we custom defined the architecture set as a constant
and thus if the user supplied a valid architecture, then without fail, it couldn't do anything other than return a valid
architecture.

reverseHot():
This function is supposed to take an array of integers, each of which corresponds to an label in the specified
classes array. In turn, this means two things need to be expected as arguments. First, the label_numpy array
must obviously be a numpy array, but it also must be a numpy array of integers. Furthermore, if the classes array
is not a list of strings, then that would also be invalid input. If either of these occur, we would return "None".
However, if not, then we expect that the label string returned would be the concatenation of the elements in the classes
array for each element that has a corresponding index.

# Sam's module and functional testing
I tested the rainforest module, which serves as the main driver of our program. For nearly all of the major features in this program. 