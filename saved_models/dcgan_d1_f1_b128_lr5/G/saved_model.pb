��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02v2.1.0-0-ge5bf8de4108��
}
dense_101/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*!
shared_namedense_101/kernel
v
$dense_101/kernel/Read/ReadVariableOpReadVariableOpdense_101/kernel*
_output_shapes
:	d�*
dtype0
u
dense_101/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_101/bias
n
"dense_101/bias/Read/ReadVariableOpReadVariableOpdense_101/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_210/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namebatch_normalization_210/gamma
�
1batch_normalization_210/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_210/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_210/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_210/beta
�
0batch_normalization_210/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_210/beta*
_output_shapes	
:�*
dtype0
�
#batch_normalization_210/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_210/moving_mean
�
7batch_normalization_210/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_210/moving_mean*
_output_shapes	
:�*
dtype0
�
'batch_normalization_210/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'batch_normalization_210/moving_variance
�
;batch_normalization_210/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_210/moving_variance*
_output_shapes	
:�*
dtype0
~
dense_102/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��b*!
shared_namedense_102/kernel
w
$dense_102/kernel/Read/ReadVariableOpReadVariableOpdense_102/kernel* 
_output_shapes
:
��b*
dtype0
u
dense_102/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�b*
shared_namedense_102/bias
n
"dense_102/bias/Read/ReadVariableOpReadVariableOpdense_102/bias*
_output_shapes	
:�b*
dtype0
�
batch_normalization_211/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�b*.
shared_namebatch_normalization_211/gamma
�
1batch_normalization_211/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_211/gamma*
_output_shapes	
:�b*
dtype0
�
batch_normalization_211/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�b*-
shared_namebatch_normalization_211/beta
�
0batch_normalization_211/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_211/beta*
_output_shapes	
:�b*
dtype0
�
#batch_normalization_211/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�b*4
shared_name%#batch_normalization_211/moving_mean
�
7batch_normalization_211/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_211/moving_mean*
_output_shapes	
:�b*
dtype0
�
'batch_normalization_211/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�b*8
shared_name)'batch_normalization_211/moving_variance
�
;batch_normalization_211/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_211/moving_variance*
_output_shapes	
:�b*
dtype0
�
conv2d_transpose_78/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameconv2d_transpose_78/kernel
�
.conv2d_transpose_78/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_78/kernel*(
_output_shapes
:��*
dtype0
�
batch_normalization_212/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namebatch_normalization_212/gamma
�
1batch_normalization_212/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_212/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_212/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_212/beta
�
0batch_normalization_212/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_212/beta*
_output_shapes	
:�*
dtype0
�
#batch_normalization_212/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_212/moving_mean
�
7batch_normalization_212/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_212/moving_mean*
_output_shapes	
:�*
dtype0
�
'batch_normalization_212/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'batch_normalization_212/moving_variance
�
;batch_normalization_212/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_212/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_79/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*+
shared_nameconv2d_transpose_79/kernel
�
.conv2d_transpose_79/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_79/kernel*'
_output_shapes
:@�*
dtype0
�
batch_normalization_213/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_213/gamma
�
1batch_normalization_213/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_213/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_213/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_213/beta
�
0batch_normalization_213/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_213/beta*
_output_shapes
:@*
dtype0
�
#batch_normalization_213/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization_213/moving_mean
�
7batch_normalization_213/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_213/moving_mean*
_output_shapes
:@*
dtype0
�
'batch_normalization_213/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'batch_normalization_213/moving_variance
�
;batch_normalization_213/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_213/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_transpose_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameconv2d_transpose_80/kernel
�
.conv2d_transpose_80/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_80/kernel*&
_output_shapes
:@*
dtype0

NoOpNoOp
�>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�>
value�>B�> B�>
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
�
axis
	gamma
beta
moving_mean
moving_variance
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
�
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4regularization_losses
5trainable_variables
6	keras_api
R
7	variables
8regularization_losses
9trainable_variables
:	keras_api
R
;	variables
<regularization_losses
=trainable_variables
>	keras_api
^

?kernel
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
�
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
R
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
^

Qkernel
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
�
Vaxis
	Wgamma
Xbeta
Ymoving_mean
Zmoving_variance
[	variables
\regularization_losses
]trainable_variables
^	keras_api
R
_	variables
`regularization_losses
atrainable_variables
b	keras_api
^

ckernel
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
�
0
1
2
3
4
5
(6
)7
/8
09
110
211
?12
E13
F14
G15
H16
Q17
W18
X19
Y20
Z21
c22
n
0
1
2
3
(4
)5
/6
07
?8
E9
F10
Q11
W12
X13
c14
 
�
	variables
hmetrics
trainable_variables

ilayers
regularization_losses
jnon_trainable_variables
klayer_regularization_losses
 
\Z
VARIABLE_VALUEdense_101/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_101/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
	variables
lmetrics
regularization_losses

mlayers
trainable_variables
nnon_trainable_variables
olayer_regularization_losses
 
hf
VARIABLE_VALUEbatch_normalization_210/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_210/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_210/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_210/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

0
1
�
 	variables
pmetrics
!regularization_losses

qlayers
"trainable_variables
rnon_trainable_variables
slayer_regularization_losses
 
 
 
�
$	variables
tmetrics
%regularization_losses

ulayers
&trainable_variables
vnon_trainable_variables
wlayer_regularization_losses
\Z
VARIABLE_VALUEdense_102/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_102/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
�
*	variables
xmetrics
+regularization_losses

ylayers
,trainable_variables
znon_trainable_variables
{layer_regularization_losses
 
hf
VARIABLE_VALUEbatch_normalization_211/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_211/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_211/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_211/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

/0
01
12
23
 

/0
01
�
3	variables
|metrics
4regularization_losses

}layers
5trainable_variables
~non_trainable_variables
layer_regularization_losses
 
 
 
�
7	variables
�metrics
8regularization_losses
�layers
9trainable_variables
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
;	variables
�metrics
<regularization_losses
�layers
=trainable_variables
�non_trainable_variables
 �layer_regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_78/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

?0
 

?0
�
@	variables
�metrics
Aregularization_losses
�layers
Btrainable_variables
�non_trainable_variables
 �layer_regularization_losses
 
hf
VARIABLE_VALUEbatch_normalization_212/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_212/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_212/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_212/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

E0
F1
G2
H3
 

E0
F1
�
I	variables
�metrics
Jregularization_losses
�layers
Ktrainable_variables
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
M	variables
�metrics
Nregularization_losses
�layers
Otrainable_variables
�non_trainable_variables
 �layer_regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_79/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE

Q0
 

Q0
�
R	variables
�metrics
Sregularization_losses
�layers
Ttrainable_variables
�non_trainable_variables
 �layer_regularization_losses
 
hf
VARIABLE_VALUEbatch_normalization_213/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_213/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_213/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_213/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
Y2
Z3
 

W0
X1
�
[	variables
�metrics
\regularization_losses
�layers
]trainable_variables
�non_trainable_variables
 �layer_regularization_losses
 
 
 
�
_	variables
�metrics
`regularization_losses
�layers
atrainable_variables
�non_trainable_variables
 �layer_regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_80/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE

c0
 

c0
�
d	variables
�metrics
eregularization_losses
�layers
ftrainable_variables
�non_trainable_variables
 �layer_regularization_losses
 
f
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
8
0
1
12
23
G4
H5
Y6
Z7
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 

10
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

G0
H1
 
 
 
 
 
 
 
 
 
 
 

Y0
Z1
 
 
 
 
 
 
 
 
 
�
serving_default_dense_101_inputPlaceholder*'
_output_shapes
:���������d*
dtype0*
shape:���������d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_101_inputdense_101/kerneldense_101/bias'batch_normalization_210/moving_variancebatch_normalization_210/gamma#batch_normalization_210/moving_meanbatch_normalization_210/betadense_102/kerneldense_102/bias'batch_normalization_211/moving_variancebatch_normalization_211/gamma#batch_normalization_211/moving_meanbatch_normalization_211/betaconv2d_transpose_78/kernelbatch_normalization_212/gammabatch_normalization_212/beta#batch_normalization_212/moving_mean'batch_normalization_212/moving_varianceconv2d_transpose_79/kernelbatch_normalization_213/gammabatch_normalization_213/beta#batch_normalization_213/moving_mean'batch_normalization_213/moving_varianceconv2d_transpose_80/kernel*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*-
config_proto

GPU

CPU2*0J 8*-
f(R&
$__inference_signature_wrapper_839365
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_101/kernel/Read/ReadVariableOp"dense_101/bias/Read/ReadVariableOp1batch_normalization_210/gamma/Read/ReadVariableOp0batch_normalization_210/beta/Read/ReadVariableOp7batch_normalization_210/moving_mean/Read/ReadVariableOp;batch_normalization_210/moving_variance/Read/ReadVariableOp$dense_102/kernel/Read/ReadVariableOp"dense_102/bias/Read/ReadVariableOp1batch_normalization_211/gamma/Read/ReadVariableOp0batch_normalization_211/beta/Read/ReadVariableOp7batch_normalization_211/moving_mean/Read/ReadVariableOp;batch_normalization_211/moving_variance/Read/ReadVariableOp.conv2d_transpose_78/kernel/Read/ReadVariableOp1batch_normalization_212/gamma/Read/ReadVariableOp0batch_normalization_212/beta/Read/ReadVariableOp7batch_normalization_212/moving_mean/Read/ReadVariableOp;batch_normalization_212/moving_variance/Read/ReadVariableOp.conv2d_transpose_79/kernel/Read/ReadVariableOp1batch_normalization_213/gamma/Read/ReadVariableOp0batch_normalization_213/beta/Read/ReadVariableOp7batch_normalization_213/moving_mean/Read/ReadVariableOp;batch_normalization_213/moving_variance/Read/ReadVariableOp.conv2d_transpose_80/kernel/Read/ReadVariableOpConst*$
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*(
f#R!
__inference__traced_save_840433
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_101/kerneldense_101/biasbatch_normalization_210/gammabatch_normalization_210/beta#batch_normalization_210/moving_mean'batch_normalization_210/moving_variancedense_102/kerneldense_102/biasbatch_normalization_211/gammabatch_normalization_211/beta#batch_normalization_211/moving_mean'batch_normalization_211/moving_varianceconv2d_transpose_78/kernelbatch_normalization_212/gammabatch_normalization_212/beta#batch_normalization_212/moving_mean'batch_normalization_212/moving_varianceconv2d_transpose_79/kernelbatch_normalization_213/gammabatch_normalization_213/beta#batch_normalization_213/moving_mean'batch_normalization_213/moving_varianceconv2d_transpose_80/kernel*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__traced_restore_840514��
�
�
E__inference_dense_101_layer_call_and_return_conditional_losses_838853

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_207_layer_call_and_return_conditional_losses_839981

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:����������2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_210_layer_call_fn_839976

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_8383092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�R
�
I__inference_sequential_53_layer_call_and_return_conditional_losses_839214

inputs,
(dense_101_statefulpartitionedcall_args_1,
(dense_101_statefulpartitionedcall_args_2:
6batch_normalization_210_statefulpartitionedcall_args_1:
6batch_normalization_210_statefulpartitionedcall_args_2:
6batch_normalization_210_statefulpartitionedcall_args_3:
6batch_normalization_210_statefulpartitionedcall_args_4,
(dense_102_statefulpartitionedcall_args_1,
(dense_102_statefulpartitionedcall_args_2:
6batch_normalization_211_statefulpartitionedcall_args_1:
6batch_normalization_211_statefulpartitionedcall_args_2:
6batch_normalization_211_statefulpartitionedcall_args_3:
6batch_normalization_211_statefulpartitionedcall_args_46
2conv2d_transpose_78_statefulpartitionedcall_args_1:
6batch_normalization_212_statefulpartitionedcall_args_1:
6batch_normalization_212_statefulpartitionedcall_args_2:
6batch_normalization_212_statefulpartitionedcall_args_3:
6batch_normalization_212_statefulpartitionedcall_args_46
2conv2d_transpose_79_statefulpartitionedcall_args_1:
6batch_normalization_213_statefulpartitionedcall_args_1:
6batch_normalization_213_statefulpartitionedcall_args_2:
6batch_normalization_213_statefulpartitionedcall_args_3:
6batch_normalization_213_statefulpartitionedcall_args_46
2conv2d_transpose_80_statefulpartitionedcall_args_1
identity��/batch_normalization_210/StatefulPartitionedCall�/batch_normalization_211/StatefulPartitionedCall�/batch_normalization_212/StatefulPartitionedCall�/batch_normalization_213/StatefulPartitionedCall�+conv2d_transpose_78/StatefulPartitionedCall�+conv2d_transpose_79/StatefulPartitionedCall�+conv2d_transpose_80/StatefulPartitionedCall�!dense_101/StatefulPartitionedCall�!dense_102/StatefulPartitionedCall�
!dense_101/StatefulPartitionedCallStatefulPartitionedCallinputs(dense_101_statefulpartitionedcall_args_1(dense_101_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_101_layer_call_and_return_conditional_losses_8388532#
!dense_101/StatefulPartitionedCall�
/batch_normalization_210/StatefulPartitionedCallStatefulPartitionedCall*dense_101/StatefulPartitionedCall:output:06batch_normalization_210_statefulpartitionedcall_args_16batch_normalization_210_statefulpartitionedcall_args_26batch_normalization_210_statefulpartitionedcall_args_36batch_normalization_210_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_83830921
/batch_normalization_210/StatefulPartitionedCall�
leaky_re_lu_207/PartitionedCallPartitionedCall8batch_normalization_210/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_207_layer_call_and_return_conditional_losses_8388932!
leaky_re_lu_207/PartitionedCall�
!dense_102/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_207/PartitionedCall:output:0(dense_102_statefulpartitionedcall_args_1(dense_102_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_102_layer_call_and_return_conditional_losses_8389112#
!dense_102/StatefulPartitionedCall�
/batch_normalization_211/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:06batch_normalization_211_statefulpartitionedcall_args_16batch_normalization_211_statefulpartitionedcall_args_26batch_normalization_211_statefulpartitionedcall_args_36batch_normalization_211_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_83845321
/batch_normalization_211/StatefulPartitionedCall�
leaky_re_lu_208/PartitionedCallPartitionedCall8batch_normalization_211/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_208_layer_call_and_return_conditional_losses_8389512!
leaky_re_lu_208/PartitionedCall�
reshape_26/PartitionedCallPartitionedCall(leaky_re_lu_208/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_reshape_26_layer_call_and_return_conditional_losses_8389732
reshape_26/PartitionedCall�
+conv2d_transpose_78/StatefulPartitionedCallStatefulPartitionedCall#reshape_26/PartitionedCall:output:02conv2d_transpose_78_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_78_layer_call_and_return_conditional_losses_8384912-
+conv2d_transpose_78/StatefulPartitionedCall�
/batch_normalization_212/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_78/StatefulPartitionedCall:output:06batch_normalization_212_statefulpartitionedcall_args_16batch_normalization_212_statefulpartitionedcall_args_26batch_normalization_212_statefulpartitionedcall_args_36batch_normalization_212_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_83862321
/batch_normalization_212/StatefulPartitionedCall�
leaky_re_lu_209/PartitionedCallPartitionedCall8batch_normalization_212/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_209_layer_call_and_return_conditional_losses_8390112!
leaky_re_lu_209/PartitionedCall�
+conv2d_transpose_79/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_209/PartitionedCall:output:02conv2d_transpose_79_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_79_layer_call_and_return_conditional_losses_8386612-
+conv2d_transpose_79/StatefulPartitionedCall�
/batch_normalization_213/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_79/StatefulPartitionedCall:output:06batch_normalization_213_statefulpartitionedcall_args_16batch_normalization_213_statefulpartitionedcall_args_26batch_normalization_213_statefulpartitionedcall_args_36batch_normalization_213_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_83879321
/batch_normalization_213/StatefulPartitionedCall�
leaky_re_lu_210/PartitionedCallPartitionedCall8batch_normalization_213/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_210_layer_call_and_return_conditional_losses_8390492!
leaky_re_lu_210/PartitionedCall�
+conv2d_transpose_80/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_210/PartitionedCall:output:02conv2d_transpose_80_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_8388322-
+conv2d_transpose_80/StatefulPartitionedCall�
IdentityIdentity4conv2d_transpose_80/StatefulPartitionedCall:output:00^batch_normalization_210/StatefulPartitionedCall0^batch_normalization_211/StatefulPartitionedCall0^batch_normalization_212/StatefulPartitionedCall0^batch_normalization_213/StatefulPartitionedCall,^conv2d_transpose_78/StatefulPartitionedCall,^conv2d_transpose_79/StatefulPartitionedCall,^conv2d_transpose_80/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::2b
/batch_normalization_210/StatefulPartitionedCall/batch_normalization_210/StatefulPartitionedCall2b
/batch_normalization_211/StatefulPartitionedCall/batch_normalization_211/StatefulPartitionedCall2b
/batch_normalization_212/StatefulPartitionedCall/batch_normalization_212/StatefulPartitionedCall2b
/batch_normalization_213/StatefulPartitionedCall/batch_normalization_213/StatefulPartitionedCall2Z
+conv2d_transpose_78/StatefulPartitionedCall+conv2d_transpose_78/StatefulPartitionedCall2Z
+conv2d_transpose_79/StatefulPartitionedCall+conv2d_transpose_79/StatefulPartitionedCall2Z
+conv2d_transpose_80/StatefulPartitionedCall+conv2d_transpose_80/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_840194

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_840179
assignmovingavg_1_840186
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/840179*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/840179*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_840179*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/840179*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/840179*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_840179AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/840179*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/840186*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/840186*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_840186*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/840186*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/840186*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_840186AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/840186*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
b
F__inference_reshape_26_layer_call_and_return_conditional_losses_838973

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :�2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:����������2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������b:& "
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_207_layer_call_and_return_conditional_losses_838893

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:����������2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
O__inference_conv2d_transpose_78_layer_call_and_return_conditional_losses_838491

inputs,
(conv2d_transpose_readvariableop_resource
identity��conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
conv2d_transpose�
IdentityIdentityconv2d_transpose:output:0 ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,����������������������������:2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_208_layer_call_and_return_conditional_losses_838951

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:����������b2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������b:& "
 
_user_specified_nameinputs
�
L
0__inference_leaky_re_lu_210_layer_call_fn_840340

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_210_layer_call_and_return_conditional_losses_8390492
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:& "
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_839365
dense_101_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_101_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__wrapped_model_8381722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_namedense_101_input
��
�
I__inference_sequential_53_layer_call_and_return_conditional_losses_839604

inputs,
(dense_101_matmul_readvariableop_resource-
)dense_101_biasadd_readvariableop_resource2
.batch_normalization_210_assignmovingavg_8393854
0batch_normalization_210_assignmovingavg_1_839391A
=batch_normalization_210_batchnorm_mul_readvariableop_resource=
9batch_normalization_210_batchnorm_readvariableop_resource,
(dense_102_matmul_readvariableop_resource-
)dense_102_biasadd_readvariableop_resource2
.batch_normalization_211_assignmovingavg_8394274
0batch_normalization_211_assignmovingavg_1_839433A
=batch_normalization_211_batchnorm_mul_readvariableop_resource=
9batch_normalization_211_batchnorm_readvariableop_resource@
<conv2d_transpose_78_conv2d_transpose_readvariableop_resource3
/batch_normalization_212_readvariableop_resource5
1batch_normalization_212_readvariableop_1_resource2
.batch_normalization_212_assignmovingavg_8395044
0batch_normalization_212_assignmovingavg_1_839511@
<conv2d_transpose_79_conv2d_transpose_readvariableop_resource3
/batch_normalization_213_readvariableop_resource5
1batch_normalization_213_readvariableop_1_resource2
.batch_normalization_213_assignmovingavg_8395614
0batch_normalization_213_assignmovingavg_1_839568@
<conv2d_transpose_80_conv2d_transpose_readvariableop_resource
identity��;batch_normalization_210/AssignMovingAvg/AssignSubVariableOp�6batch_normalization_210/AssignMovingAvg/ReadVariableOp�=batch_normalization_210/AssignMovingAvg_1/AssignSubVariableOp�8batch_normalization_210/AssignMovingAvg_1/ReadVariableOp�0batch_normalization_210/batchnorm/ReadVariableOp�4batch_normalization_210/batchnorm/mul/ReadVariableOp�;batch_normalization_211/AssignMovingAvg/AssignSubVariableOp�6batch_normalization_211/AssignMovingAvg/ReadVariableOp�=batch_normalization_211/AssignMovingAvg_1/AssignSubVariableOp�8batch_normalization_211/AssignMovingAvg_1/ReadVariableOp�0batch_normalization_211/batchnorm/ReadVariableOp�4batch_normalization_211/batchnorm/mul/ReadVariableOp�;batch_normalization_212/AssignMovingAvg/AssignSubVariableOp�6batch_normalization_212/AssignMovingAvg/ReadVariableOp�=batch_normalization_212/AssignMovingAvg_1/AssignSubVariableOp�8batch_normalization_212/AssignMovingAvg_1/ReadVariableOp�&batch_normalization_212/ReadVariableOp�(batch_normalization_212/ReadVariableOp_1�;batch_normalization_213/AssignMovingAvg/AssignSubVariableOp�6batch_normalization_213/AssignMovingAvg/ReadVariableOp�=batch_normalization_213/AssignMovingAvg_1/AssignSubVariableOp�8batch_normalization_213/AssignMovingAvg_1/ReadVariableOp�&batch_normalization_213/ReadVariableOp�(batch_normalization_213/ReadVariableOp_1�3conv2d_transpose_78/conv2d_transpose/ReadVariableOp�3conv2d_transpose_79/conv2d_transpose/ReadVariableOp�3conv2d_transpose_80/conv2d_transpose/ReadVariableOp� dense_101/BiasAdd/ReadVariableOp�dense_101/MatMul/ReadVariableOp� dense_102/BiasAdd/ReadVariableOp�dense_102/MatMul/ReadVariableOp�
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02!
dense_101/MatMul/ReadVariableOp�
dense_101/MatMulMatMulinputs'dense_101/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_101/MatMul�
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dense_101/BiasAdd/ReadVariableOp�
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_101/BiasAdd�
$batch_normalization_210/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_210/LogicalAnd/x�
$batch_normalization_210/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_210/LogicalAnd/y�
"batch_normalization_210/LogicalAnd
LogicalAnd-batch_normalization_210/LogicalAnd/x:output:0-batch_normalization_210/LogicalAnd/y:output:0*
_output_shapes
: 2$
"batch_normalization_210/LogicalAnd�
6batch_normalization_210/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_210/moments/mean/reduction_indices�
$batch_normalization_210/moments/meanMeandense_101/BiasAdd:output:0?batch_normalization_210/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2&
$batch_normalization_210/moments/mean�
,batch_normalization_210/moments/StopGradientStopGradient-batch_normalization_210/moments/mean:output:0*
T0*
_output_shapes
:	�2.
,batch_normalization_210/moments/StopGradient�
1batch_normalization_210/moments/SquaredDifferenceSquaredDifferencedense_101/BiasAdd:output:05batch_normalization_210/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������23
1batch_normalization_210/moments/SquaredDifference�
:batch_normalization_210/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_210/moments/variance/reduction_indices�
(batch_normalization_210/moments/varianceMean5batch_normalization_210/moments/SquaredDifference:z:0Cbatch_normalization_210/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2*
(batch_normalization_210/moments/variance�
'batch_normalization_210/moments/SqueezeSqueeze-batch_normalization_210/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_210/moments/Squeeze�
)batch_normalization_210/moments/Squeeze_1Squeeze1batch_normalization_210/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2+
)batch_normalization_210/moments/Squeeze_1�
-batch_normalization_210/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_210/AssignMovingAvg/839385*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_210/AssignMovingAvg/decay�
6batch_normalization_210/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_210_assignmovingavg_839385*
_output_shapes	
:�*
dtype028
6batch_normalization_210/AssignMovingAvg/ReadVariableOp�
+batch_normalization_210/AssignMovingAvg/subSub>batch_normalization_210/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_210/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_210/AssignMovingAvg/839385*
_output_shapes	
:�2-
+batch_normalization_210/AssignMovingAvg/sub�
+batch_normalization_210/AssignMovingAvg/mulMul/batch_normalization_210/AssignMovingAvg/sub:z:06batch_normalization_210/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_210/AssignMovingAvg/839385*
_output_shapes	
:�2-
+batch_normalization_210/AssignMovingAvg/mul�
;batch_normalization_210/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_210_assignmovingavg_839385/batch_normalization_210/AssignMovingAvg/mul:z:07^batch_normalization_210/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_210/AssignMovingAvg/839385*
_output_shapes
 *
dtype02=
;batch_normalization_210/AssignMovingAvg/AssignSubVariableOp�
/batch_normalization_210/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_210/AssignMovingAvg_1/839391*
_output_shapes
: *
dtype0*
valueB
 *
�#<21
/batch_normalization_210/AssignMovingAvg_1/decay�
8batch_normalization_210/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_210_assignmovingavg_1_839391*
_output_shapes	
:�*
dtype02:
8batch_normalization_210/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_210/AssignMovingAvg_1/subSub@batch_normalization_210/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_210/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_210/AssignMovingAvg_1/839391*
_output_shapes	
:�2/
-batch_normalization_210/AssignMovingAvg_1/sub�
-batch_normalization_210/AssignMovingAvg_1/mulMul1batch_normalization_210/AssignMovingAvg_1/sub:z:08batch_normalization_210/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_210/AssignMovingAvg_1/839391*
_output_shapes	
:�2/
-batch_normalization_210/AssignMovingAvg_1/mul�
=batch_normalization_210/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_210_assignmovingavg_1_8393911batch_normalization_210/AssignMovingAvg_1/mul:z:09^batch_normalization_210/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_210/AssignMovingAvg_1/839391*
_output_shapes
 *
dtype02?
=batch_normalization_210/AssignMovingAvg_1/AssignSubVariableOp�
'batch_normalization_210/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2)
'batch_normalization_210/batchnorm/add/y�
%batch_normalization_210/batchnorm/addAddV22batch_normalization_210/moments/Squeeze_1:output:00batch_normalization_210/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2'
%batch_normalization_210/batchnorm/add�
'batch_normalization_210/batchnorm/RsqrtRsqrt)batch_normalization_210/batchnorm/add:z:0*
T0*
_output_shapes	
:�2)
'batch_normalization_210/batchnorm/Rsqrt�
4batch_normalization_210/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_210_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_210/batchnorm/mul/ReadVariableOp�
%batch_normalization_210/batchnorm/mulMul+batch_normalization_210/batchnorm/Rsqrt:y:0<batch_normalization_210/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2'
%batch_normalization_210/batchnorm/mul�
'batch_normalization_210/batchnorm/mul_1Muldense_101/BiasAdd:output:0)batch_normalization_210/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2)
'batch_normalization_210/batchnorm/mul_1�
'batch_normalization_210/batchnorm/mul_2Mul0batch_normalization_210/moments/Squeeze:output:0)batch_normalization_210/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2)
'batch_normalization_210/batchnorm/mul_2�
0batch_normalization_210/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_210_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_210/batchnorm/ReadVariableOp�
%batch_normalization_210/batchnorm/subSub8batch_normalization_210/batchnorm/ReadVariableOp:value:0+batch_normalization_210/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_210/batchnorm/sub�
'batch_normalization_210/batchnorm/add_1AddV2+batch_normalization_210/batchnorm/mul_1:z:0)batch_normalization_210/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2)
'batch_normalization_210/batchnorm/add_1�
leaky_re_lu_207/LeakyRelu	LeakyRelu+batch_normalization_210/batchnorm/add_1:z:0*(
_output_shapes
:����������2
leaky_re_lu_207/LeakyRelu�
dense_102/MatMul/ReadVariableOpReadVariableOp(dense_102_matmul_readvariableop_resource* 
_output_shapes
:
��b*
dtype02!
dense_102/MatMul/ReadVariableOp�
dense_102/MatMulMatMul'leaky_re_lu_207/LeakyRelu:activations:0'dense_102/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b2
dense_102/MatMul�
 dense_102/BiasAdd/ReadVariableOpReadVariableOp)dense_102_biasadd_readvariableop_resource*
_output_shapes	
:�b*
dtype02"
 dense_102/BiasAdd/ReadVariableOp�
dense_102/BiasAddBiasAdddense_102/MatMul:product:0(dense_102/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b2
dense_102/BiasAdd�
$batch_normalization_211/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_211/LogicalAnd/x�
$batch_normalization_211/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_211/LogicalAnd/y�
"batch_normalization_211/LogicalAnd
LogicalAnd-batch_normalization_211/LogicalAnd/x:output:0-batch_normalization_211/LogicalAnd/y:output:0*
_output_shapes
: 2$
"batch_normalization_211/LogicalAnd�
6batch_normalization_211/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_211/moments/mean/reduction_indices�
$batch_normalization_211/moments/meanMeandense_102/BiasAdd:output:0?batch_normalization_211/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�b*
	keep_dims(2&
$batch_normalization_211/moments/mean�
,batch_normalization_211/moments/StopGradientStopGradient-batch_normalization_211/moments/mean:output:0*
T0*
_output_shapes
:	�b2.
,batch_normalization_211/moments/StopGradient�
1batch_normalization_211/moments/SquaredDifferenceSquaredDifferencedense_102/BiasAdd:output:05batch_normalization_211/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������b23
1batch_normalization_211/moments/SquaredDifference�
:batch_normalization_211/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_211/moments/variance/reduction_indices�
(batch_normalization_211/moments/varianceMean5batch_normalization_211/moments/SquaredDifference:z:0Cbatch_normalization_211/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�b*
	keep_dims(2*
(batch_normalization_211/moments/variance�
'batch_normalization_211/moments/SqueezeSqueeze-batch_normalization_211/moments/mean:output:0*
T0*
_output_shapes	
:�b*
squeeze_dims
 2)
'batch_normalization_211/moments/Squeeze�
)batch_normalization_211/moments/Squeeze_1Squeeze1batch_normalization_211/moments/variance:output:0*
T0*
_output_shapes	
:�b*
squeeze_dims
 2+
)batch_normalization_211/moments/Squeeze_1�
-batch_normalization_211/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_211/AssignMovingAvg/839427*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_211/AssignMovingAvg/decay�
6batch_normalization_211/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_211_assignmovingavg_839427*
_output_shapes	
:�b*
dtype028
6batch_normalization_211/AssignMovingAvg/ReadVariableOp�
+batch_normalization_211/AssignMovingAvg/subSub>batch_normalization_211/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_211/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_211/AssignMovingAvg/839427*
_output_shapes	
:�b2-
+batch_normalization_211/AssignMovingAvg/sub�
+batch_normalization_211/AssignMovingAvg/mulMul/batch_normalization_211/AssignMovingAvg/sub:z:06batch_normalization_211/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_211/AssignMovingAvg/839427*
_output_shapes	
:�b2-
+batch_normalization_211/AssignMovingAvg/mul�
;batch_normalization_211/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_211_assignmovingavg_839427/batch_normalization_211/AssignMovingAvg/mul:z:07^batch_normalization_211/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_211/AssignMovingAvg/839427*
_output_shapes
 *
dtype02=
;batch_normalization_211/AssignMovingAvg/AssignSubVariableOp�
/batch_normalization_211/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_211/AssignMovingAvg_1/839433*
_output_shapes
: *
dtype0*
valueB
 *
�#<21
/batch_normalization_211/AssignMovingAvg_1/decay�
8batch_normalization_211/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_211_assignmovingavg_1_839433*
_output_shapes	
:�b*
dtype02:
8batch_normalization_211/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_211/AssignMovingAvg_1/subSub@batch_normalization_211/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_211/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_211/AssignMovingAvg_1/839433*
_output_shapes	
:�b2/
-batch_normalization_211/AssignMovingAvg_1/sub�
-batch_normalization_211/AssignMovingAvg_1/mulMul1batch_normalization_211/AssignMovingAvg_1/sub:z:08batch_normalization_211/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_211/AssignMovingAvg_1/839433*
_output_shapes	
:�b2/
-batch_normalization_211/AssignMovingAvg_1/mul�
=batch_normalization_211/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_211_assignmovingavg_1_8394331batch_normalization_211/AssignMovingAvg_1/mul:z:09^batch_normalization_211/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_211/AssignMovingAvg_1/839433*
_output_shapes
 *
dtype02?
=batch_normalization_211/AssignMovingAvg_1/AssignSubVariableOp�
'batch_normalization_211/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2)
'batch_normalization_211/batchnorm/add/y�
%batch_normalization_211/batchnorm/addAddV22batch_normalization_211/moments/Squeeze_1:output:00batch_normalization_211/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�b2'
%batch_normalization_211/batchnorm/add�
'batch_normalization_211/batchnorm/RsqrtRsqrt)batch_normalization_211/batchnorm/add:z:0*
T0*
_output_shapes	
:�b2)
'batch_normalization_211/batchnorm/Rsqrt�
4batch_normalization_211/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_211_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�b*
dtype026
4batch_normalization_211/batchnorm/mul/ReadVariableOp�
%batch_normalization_211/batchnorm/mulMul+batch_normalization_211/batchnorm/Rsqrt:y:0<batch_normalization_211/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b2'
%batch_normalization_211/batchnorm/mul�
'batch_normalization_211/batchnorm/mul_1Muldense_102/BiasAdd:output:0)batch_normalization_211/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������b2)
'batch_normalization_211/batchnorm/mul_1�
'batch_normalization_211/batchnorm/mul_2Mul0batch_normalization_211/moments/Squeeze:output:0)batch_normalization_211/batchnorm/mul:z:0*
T0*
_output_shapes	
:�b2)
'batch_normalization_211/batchnorm/mul_2�
0batch_normalization_211/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_211_batchnorm_readvariableop_resource*
_output_shapes	
:�b*
dtype022
0batch_normalization_211/batchnorm/ReadVariableOp�
%batch_normalization_211/batchnorm/subSub8batch_normalization_211/batchnorm/ReadVariableOp:value:0+batch_normalization_211/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�b2'
%batch_normalization_211/batchnorm/sub�
'batch_normalization_211/batchnorm/add_1AddV2+batch_normalization_211/batchnorm/mul_1:z:0)batch_normalization_211/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������b2)
'batch_normalization_211/batchnorm/add_1�
leaky_re_lu_208/LeakyRelu	LeakyRelu+batch_normalization_211/batchnorm/add_1:z:0*(
_output_shapes
:����������b2
leaky_re_lu_208/LeakyRelu{
reshape_26/ShapeShape'leaky_re_lu_208/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_26/Shape�
reshape_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_26/strided_slice/stack�
 reshape_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_26/strided_slice/stack_1�
 reshape_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_26/strided_slice/stack_2�
reshape_26/strided_sliceStridedSlicereshape_26/Shape:output:0'reshape_26/strided_slice/stack:output:0)reshape_26/strided_slice/stack_1:output:0)reshape_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_26/strided_slicez
reshape_26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_26/Reshape/shape/1z
reshape_26/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_26/Reshape/shape/2{
reshape_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :�2
reshape_26/Reshape/shape/3�
reshape_26/Reshape/shapePack!reshape_26/strided_slice:output:0#reshape_26/Reshape/shape/1:output:0#reshape_26/Reshape/shape/2:output:0#reshape_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_26/Reshape/shape�
reshape_26/ReshapeReshape'leaky_re_lu_208/LeakyRelu:activations:0!reshape_26/Reshape/shape:output:0*
T0*0
_output_shapes
:����������2
reshape_26/Reshape�
conv2d_transpose_78/ShapeShapereshape_26/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_78/Shape�
'conv2d_transpose_78/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_78/strided_slice/stack�
)conv2d_transpose_78/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_78/strided_slice/stack_1�
)conv2d_transpose_78/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_78/strided_slice/stack_2�
!conv2d_transpose_78/strided_sliceStridedSlice"conv2d_transpose_78/Shape:output:00conv2d_transpose_78/strided_slice/stack:output:02conv2d_transpose_78/strided_slice/stack_1:output:02conv2d_transpose_78/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_78/strided_slice�
)conv2d_transpose_78/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_78/strided_slice_1/stack�
+conv2d_transpose_78/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_1/stack_1�
+conv2d_transpose_78/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_1/stack_2�
#conv2d_transpose_78/strided_slice_1StridedSlice"conv2d_transpose_78/Shape:output:02conv2d_transpose_78/strided_slice_1/stack:output:04conv2d_transpose_78/strided_slice_1/stack_1:output:04conv2d_transpose_78/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_78/strided_slice_1�
)conv2d_transpose_78/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_78/strided_slice_2/stack�
+conv2d_transpose_78/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_2/stack_1�
+conv2d_transpose_78/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_2/stack_2�
#conv2d_transpose_78/strided_slice_2StridedSlice"conv2d_transpose_78/Shape:output:02conv2d_transpose_78/strided_slice_2/stack:output:04conv2d_transpose_78/strided_slice_2/stack_1:output:04conv2d_transpose_78/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_78/strided_slice_2x
conv2d_transpose_78/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_78/mul/y�
conv2d_transpose_78/mulMul,conv2d_transpose_78/strided_slice_1:output:0"conv2d_transpose_78/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_78/mul|
conv2d_transpose_78/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_78/mul_1/y�
conv2d_transpose_78/mul_1Mul,conv2d_transpose_78/strided_slice_2:output:0$conv2d_transpose_78/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_78/mul_1}
conv2d_transpose_78/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
conv2d_transpose_78/stack/3�
conv2d_transpose_78/stackPack*conv2d_transpose_78/strided_slice:output:0conv2d_transpose_78/mul:z:0conv2d_transpose_78/mul_1:z:0$conv2d_transpose_78/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_78/stack�
)conv2d_transpose_78/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_78/strided_slice_3/stack�
+conv2d_transpose_78/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_3/stack_1�
+conv2d_transpose_78/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_3/stack_2�
#conv2d_transpose_78/strided_slice_3StridedSlice"conv2d_transpose_78/stack:output:02conv2d_transpose_78/strided_slice_3/stack:output:04conv2d_transpose_78/strided_slice_3/stack_1:output:04conv2d_transpose_78/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_78/strided_slice_3�
3conv2d_transpose_78/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_78_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype025
3conv2d_transpose_78/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_78/conv2d_transposeConv2DBackpropInput"conv2d_transpose_78/stack:output:0;conv2d_transpose_78/conv2d_transpose/ReadVariableOp:value:0reshape_26/Reshape:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$conv2d_transpose_78/conv2d_transpose�
$batch_normalization_212/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_212/LogicalAnd/x�
$batch_normalization_212/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_212/LogicalAnd/y�
"batch_normalization_212/LogicalAnd
LogicalAnd-batch_normalization_212/LogicalAnd/x:output:0-batch_normalization_212/LogicalAnd/y:output:0*
_output_shapes
: 2$
"batch_normalization_212/LogicalAnd�
&batch_normalization_212/ReadVariableOpReadVariableOp/batch_normalization_212_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_212/ReadVariableOp�
(batch_normalization_212/ReadVariableOp_1ReadVariableOp1batch_normalization_212_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_212/ReadVariableOp_1�
batch_normalization_212/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_212/Const�
batch_normalization_212/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2!
batch_normalization_212/Const_1�
(batch_normalization_212/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_78/conv2d_transpose:output:0.batch_normalization_212/ReadVariableOp:value:00batch_normalization_212/ReadVariableOp_1:value:0&batch_normalization_212/Const:output:0(batch_normalization_212/Const_1:output:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:2*
(batch_normalization_212/FusedBatchNormV3�
batch_normalization_212/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2!
batch_normalization_212/Const_2�
-batch_normalization_212/AssignMovingAvg/sub/xConst*A
_class7
53loc:@batch_normalization_212/AssignMovingAvg/839504*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_212/AssignMovingAvg/sub/x�
+batch_normalization_212/AssignMovingAvg/subSub6batch_normalization_212/AssignMovingAvg/sub/x:output:0(batch_normalization_212/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_212/AssignMovingAvg/839504*
_output_shapes
: 2-
+batch_normalization_212/AssignMovingAvg/sub�
6batch_normalization_212/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_212_assignmovingavg_839504*
_output_shapes	
:�*
dtype028
6batch_normalization_212/AssignMovingAvg/ReadVariableOp�
-batch_normalization_212/AssignMovingAvg/sub_1Sub>batch_normalization_212/AssignMovingAvg/ReadVariableOp:value:05batch_normalization_212/FusedBatchNormV3:batch_mean:0*
T0*A
_class7
53loc:@batch_normalization_212/AssignMovingAvg/839504*
_output_shapes	
:�2/
-batch_normalization_212/AssignMovingAvg/sub_1�
+batch_normalization_212/AssignMovingAvg/mulMul1batch_normalization_212/AssignMovingAvg/sub_1:z:0/batch_normalization_212/AssignMovingAvg/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_212/AssignMovingAvg/839504*
_output_shapes	
:�2-
+batch_normalization_212/AssignMovingAvg/mul�
;batch_normalization_212/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_212_assignmovingavg_839504/batch_normalization_212/AssignMovingAvg/mul:z:07^batch_normalization_212/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_212/AssignMovingAvg/839504*
_output_shapes
 *
dtype02=
;batch_normalization_212/AssignMovingAvg/AssignSubVariableOp�
/batch_normalization_212/AssignMovingAvg_1/sub/xConst*C
_class9
75loc:@batch_normalization_212/AssignMovingAvg_1/839511*
_output_shapes
: *
dtype0*
valueB
 *  �?21
/batch_normalization_212/AssignMovingAvg_1/sub/x�
-batch_normalization_212/AssignMovingAvg_1/subSub8batch_normalization_212/AssignMovingAvg_1/sub/x:output:0(batch_normalization_212/Const_2:output:0*
T0*C
_class9
75loc:@batch_normalization_212/AssignMovingAvg_1/839511*
_output_shapes
: 2/
-batch_normalization_212/AssignMovingAvg_1/sub�
8batch_normalization_212/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_212_assignmovingavg_1_839511*
_output_shapes	
:�*
dtype02:
8batch_normalization_212/AssignMovingAvg_1/ReadVariableOp�
/batch_normalization_212/AssignMovingAvg_1/sub_1Sub@batch_normalization_212/AssignMovingAvg_1/ReadVariableOp:value:09batch_normalization_212/FusedBatchNormV3:batch_variance:0*
T0*C
_class9
75loc:@batch_normalization_212/AssignMovingAvg_1/839511*
_output_shapes	
:�21
/batch_normalization_212/AssignMovingAvg_1/sub_1�
-batch_normalization_212/AssignMovingAvg_1/mulMul3batch_normalization_212/AssignMovingAvg_1/sub_1:z:01batch_normalization_212/AssignMovingAvg_1/sub:z:0*
T0*C
_class9
75loc:@batch_normalization_212/AssignMovingAvg_1/839511*
_output_shapes	
:�2/
-batch_normalization_212/AssignMovingAvg_1/mul�
=batch_normalization_212/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_212_assignmovingavg_1_8395111batch_normalization_212/AssignMovingAvg_1/mul:z:09^batch_normalization_212/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_212/AssignMovingAvg_1/839511*
_output_shapes
 *
dtype02?
=batch_normalization_212/AssignMovingAvg_1/AssignSubVariableOp�
leaky_re_lu_209/LeakyRelu	LeakyRelu,batch_normalization_212/FusedBatchNormV3:y:0*0
_output_shapes
:����������2
leaky_re_lu_209/LeakyRelu�
conv2d_transpose_79/ShapeShape'leaky_re_lu_209/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_79/Shape�
'conv2d_transpose_79/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_79/strided_slice/stack�
)conv2d_transpose_79/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_79/strided_slice/stack_1�
)conv2d_transpose_79/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_79/strided_slice/stack_2�
!conv2d_transpose_79/strided_sliceStridedSlice"conv2d_transpose_79/Shape:output:00conv2d_transpose_79/strided_slice/stack:output:02conv2d_transpose_79/strided_slice/stack_1:output:02conv2d_transpose_79/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_79/strided_slice�
)conv2d_transpose_79/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_79/strided_slice_1/stack�
+conv2d_transpose_79/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_1/stack_1�
+conv2d_transpose_79/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_1/stack_2�
#conv2d_transpose_79/strided_slice_1StridedSlice"conv2d_transpose_79/Shape:output:02conv2d_transpose_79/strided_slice_1/stack:output:04conv2d_transpose_79/strided_slice_1/stack_1:output:04conv2d_transpose_79/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_79/strided_slice_1�
)conv2d_transpose_79/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_79/strided_slice_2/stack�
+conv2d_transpose_79/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_2/stack_1�
+conv2d_transpose_79/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_2/stack_2�
#conv2d_transpose_79/strided_slice_2StridedSlice"conv2d_transpose_79/Shape:output:02conv2d_transpose_79/strided_slice_2/stack:output:04conv2d_transpose_79/strided_slice_2/stack_1:output:04conv2d_transpose_79/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_79/strided_slice_2x
conv2d_transpose_79/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_79/mul/y�
conv2d_transpose_79/mulMul,conv2d_transpose_79/strided_slice_1:output:0"conv2d_transpose_79/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_79/mul|
conv2d_transpose_79/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_79/mul_1/y�
conv2d_transpose_79/mul_1Mul,conv2d_transpose_79/strided_slice_2:output:0$conv2d_transpose_79/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_79/mul_1|
conv2d_transpose_79/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_79/stack/3�
conv2d_transpose_79/stackPack*conv2d_transpose_79/strided_slice:output:0conv2d_transpose_79/mul:z:0conv2d_transpose_79/mul_1:z:0$conv2d_transpose_79/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_79/stack�
)conv2d_transpose_79/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_79/strided_slice_3/stack�
+conv2d_transpose_79/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_3/stack_1�
+conv2d_transpose_79/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_3/stack_2�
#conv2d_transpose_79/strided_slice_3StridedSlice"conv2d_transpose_79/stack:output:02conv2d_transpose_79/strided_slice_3/stack:output:04conv2d_transpose_79/strided_slice_3/stack_1:output:04conv2d_transpose_79/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_79/strided_slice_3�
3conv2d_transpose_79/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_79_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype025
3conv2d_transpose_79/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_79/conv2d_transposeConv2DBackpropInput"conv2d_transpose_79/stack:output:0;conv2d_transpose_79/conv2d_transpose/ReadVariableOp:value:0'leaky_re_lu_209/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2&
$conv2d_transpose_79/conv2d_transpose�
$batch_normalization_213/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_213/LogicalAnd/x�
$batch_normalization_213/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_213/LogicalAnd/y�
"batch_normalization_213/LogicalAnd
LogicalAnd-batch_normalization_213/LogicalAnd/x:output:0-batch_normalization_213/LogicalAnd/y:output:0*
_output_shapes
: 2$
"batch_normalization_213/LogicalAnd�
&batch_normalization_213/ReadVariableOpReadVariableOp/batch_normalization_213_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_213/ReadVariableOp�
(batch_normalization_213/ReadVariableOp_1ReadVariableOp1batch_normalization_213_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_213/ReadVariableOp_1�
batch_normalization_213/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_213/Const�
batch_normalization_213/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2!
batch_normalization_213/Const_1�
(batch_normalization_213/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_79/conv2d_transpose:output:0.batch_normalization_213/ReadVariableOp:value:00batch_normalization_213/ReadVariableOp_1:value:0&batch_normalization_213/Const:output:0(batch_normalization_213/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2*
(batch_normalization_213/FusedBatchNormV3�
batch_normalization_213/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2!
batch_normalization_213/Const_2�
-batch_normalization_213/AssignMovingAvg/sub/xConst*A
_class7
53loc:@batch_normalization_213/AssignMovingAvg/839561*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_213/AssignMovingAvg/sub/x�
+batch_normalization_213/AssignMovingAvg/subSub6batch_normalization_213/AssignMovingAvg/sub/x:output:0(batch_normalization_213/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_213/AssignMovingAvg/839561*
_output_shapes
: 2-
+batch_normalization_213/AssignMovingAvg/sub�
6batch_normalization_213/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_213_assignmovingavg_839561*
_output_shapes
:@*
dtype028
6batch_normalization_213/AssignMovingAvg/ReadVariableOp�
-batch_normalization_213/AssignMovingAvg/sub_1Sub>batch_normalization_213/AssignMovingAvg/ReadVariableOp:value:05batch_normalization_213/FusedBatchNormV3:batch_mean:0*
T0*A
_class7
53loc:@batch_normalization_213/AssignMovingAvg/839561*
_output_shapes
:@2/
-batch_normalization_213/AssignMovingAvg/sub_1�
+batch_normalization_213/AssignMovingAvg/mulMul1batch_normalization_213/AssignMovingAvg/sub_1:z:0/batch_normalization_213/AssignMovingAvg/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_213/AssignMovingAvg/839561*
_output_shapes
:@2-
+batch_normalization_213/AssignMovingAvg/mul�
;batch_normalization_213/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_213_assignmovingavg_839561/batch_normalization_213/AssignMovingAvg/mul:z:07^batch_normalization_213/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_213/AssignMovingAvg/839561*
_output_shapes
 *
dtype02=
;batch_normalization_213/AssignMovingAvg/AssignSubVariableOp�
/batch_normalization_213/AssignMovingAvg_1/sub/xConst*C
_class9
75loc:@batch_normalization_213/AssignMovingAvg_1/839568*
_output_shapes
: *
dtype0*
valueB
 *  �?21
/batch_normalization_213/AssignMovingAvg_1/sub/x�
-batch_normalization_213/AssignMovingAvg_1/subSub8batch_normalization_213/AssignMovingAvg_1/sub/x:output:0(batch_normalization_213/Const_2:output:0*
T0*C
_class9
75loc:@batch_normalization_213/AssignMovingAvg_1/839568*
_output_shapes
: 2/
-batch_normalization_213/AssignMovingAvg_1/sub�
8batch_normalization_213/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_213_assignmovingavg_1_839568*
_output_shapes
:@*
dtype02:
8batch_normalization_213/AssignMovingAvg_1/ReadVariableOp�
/batch_normalization_213/AssignMovingAvg_1/sub_1Sub@batch_normalization_213/AssignMovingAvg_1/ReadVariableOp:value:09batch_normalization_213/FusedBatchNormV3:batch_variance:0*
T0*C
_class9
75loc:@batch_normalization_213/AssignMovingAvg_1/839568*
_output_shapes
:@21
/batch_normalization_213/AssignMovingAvg_1/sub_1�
-batch_normalization_213/AssignMovingAvg_1/mulMul3batch_normalization_213/AssignMovingAvg_1/sub_1:z:01batch_normalization_213/AssignMovingAvg_1/sub:z:0*
T0*C
_class9
75loc:@batch_normalization_213/AssignMovingAvg_1/839568*
_output_shapes
:@2/
-batch_normalization_213/AssignMovingAvg_1/mul�
=batch_normalization_213/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_213_assignmovingavg_1_8395681batch_normalization_213/AssignMovingAvg_1/mul:z:09^batch_normalization_213/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_213/AssignMovingAvg_1/839568*
_output_shapes
 *
dtype02?
=batch_normalization_213/AssignMovingAvg_1/AssignSubVariableOp�
leaky_re_lu_210/LeakyRelu	LeakyRelu,batch_normalization_213/FusedBatchNormV3:y:0*/
_output_shapes
:���������@2
leaky_re_lu_210/LeakyRelu�
conv2d_transpose_80/ShapeShape'leaky_re_lu_210/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_80/Shape�
'conv2d_transpose_80/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_80/strided_slice/stack�
)conv2d_transpose_80/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice/stack_1�
)conv2d_transpose_80/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice/stack_2�
!conv2d_transpose_80/strided_sliceStridedSlice"conv2d_transpose_80/Shape:output:00conv2d_transpose_80/strided_slice/stack:output:02conv2d_transpose_80/strided_slice/stack_1:output:02conv2d_transpose_80/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_80/strided_slice�
)conv2d_transpose_80/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice_1/stack�
+conv2d_transpose_80/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_1/stack_1�
+conv2d_transpose_80/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_1/stack_2�
#conv2d_transpose_80/strided_slice_1StridedSlice"conv2d_transpose_80/Shape:output:02conv2d_transpose_80/strided_slice_1/stack:output:04conv2d_transpose_80/strided_slice_1/stack_1:output:04conv2d_transpose_80/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_80/strided_slice_1�
)conv2d_transpose_80/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice_2/stack�
+conv2d_transpose_80/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_2/stack_1�
+conv2d_transpose_80/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_2/stack_2�
#conv2d_transpose_80/strided_slice_2StridedSlice"conv2d_transpose_80/Shape:output:02conv2d_transpose_80/strided_slice_2/stack:output:04conv2d_transpose_80/strided_slice_2/stack_1:output:04conv2d_transpose_80/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_80/strided_slice_2x
conv2d_transpose_80/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_80/mul/y�
conv2d_transpose_80/mulMul,conv2d_transpose_80/strided_slice_1:output:0"conv2d_transpose_80/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_80/mul|
conv2d_transpose_80/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_80/mul_1/y�
conv2d_transpose_80/mul_1Mul,conv2d_transpose_80/strided_slice_2:output:0$conv2d_transpose_80/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_80/mul_1|
conv2d_transpose_80/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_80/stack/3�
conv2d_transpose_80/stackPack*conv2d_transpose_80/strided_slice:output:0conv2d_transpose_80/mul:z:0conv2d_transpose_80/mul_1:z:0$conv2d_transpose_80/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_80/stack�
)conv2d_transpose_80/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_80/strided_slice_3/stack�
+conv2d_transpose_80/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_3/stack_1�
+conv2d_transpose_80/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_3/stack_2�
#conv2d_transpose_80/strided_slice_3StridedSlice"conv2d_transpose_80/stack:output:02conv2d_transpose_80/strided_slice_3/stack:output:04conv2d_transpose_80/strided_slice_3/stack_1:output:04conv2d_transpose_80/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_80/strided_slice_3�
3conv2d_transpose_80/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_80_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype025
3conv2d_transpose_80/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_80/conv2d_transposeConv2DBackpropInput"conv2d_transpose_80/stack:output:0;conv2d_transpose_80/conv2d_transpose/ReadVariableOp:value:0'leaky_re_lu_210/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2&
$conv2d_transpose_80/conv2d_transpose�
conv2d_transpose_80/TanhTanh-conv2d_transpose_80/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������2
conv2d_transpose_80/Tanh�
IdentityIdentityconv2d_transpose_80/Tanh:y:0<^batch_normalization_210/AssignMovingAvg/AssignSubVariableOp7^batch_normalization_210/AssignMovingAvg/ReadVariableOp>^batch_normalization_210/AssignMovingAvg_1/AssignSubVariableOp9^batch_normalization_210/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_210/batchnorm/ReadVariableOp5^batch_normalization_210/batchnorm/mul/ReadVariableOp<^batch_normalization_211/AssignMovingAvg/AssignSubVariableOp7^batch_normalization_211/AssignMovingAvg/ReadVariableOp>^batch_normalization_211/AssignMovingAvg_1/AssignSubVariableOp9^batch_normalization_211/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_211/batchnorm/ReadVariableOp5^batch_normalization_211/batchnorm/mul/ReadVariableOp<^batch_normalization_212/AssignMovingAvg/AssignSubVariableOp7^batch_normalization_212/AssignMovingAvg/ReadVariableOp>^batch_normalization_212/AssignMovingAvg_1/AssignSubVariableOp9^batch_normalization_212/AssignMovingAvg_1/ReadVariableOp'^batch_normalization_212/ReadVariableOp)^batch_normalization_212/ReadVariableOp_1<^batch_normalization_213/AssignMovingAvg/AssignSubVariableOp7^batch_normalization_213/AssignMovingAvg/ReadVariableOp>^batch_normalization_213/AssignMovingAvg_1/AssignSubVariableOp9^batch_normalization_213/AssignMovingAvg_1/ReadVariableOp'^batch_normalization_213/ReadVariableOp)^batch_normalization_213/ReadVariableOp_14^conv2d_transpose_78/conv2d_transpose/ReadVariableOp4^conv2d_transpose_79/conv2d_transpose/ReadVariableOp4^conv2d_transpose_80/conv2d_transpose/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp!^dense_102/BiasAdd/ReadVariableOp ^dense_102/MatMul/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::2z
;batch_normalization_210/AssignMovingAvg/AssignSubVariableOp;batch_normalization_210/AssignMovingAvg/AssignSubVariableOp2p
6batch_normalization_210/AssignMovingAvg/ReadVariableOp6batch_normalization_210/AssignMovingAvg/ReadVariableOp2~
=batch_normalization_210/AssignMovingAvg_1/AssignSubVariableOp=batch_normalization_210/AssignMovingAvg_1/AssignSubVariableOp2t
8batch_normalization_210/AssignMovingAvg_1/ReadVariableOp8batch_normalization_210/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_210/batchnorm/ReadVariableOp0batch_normalization_210/batchnorm/ReadVariableOp2l
4batch_normalization_210/batchnorm/mul/ReadVariableOp4batch_normalization_210/batchnorm/mul/ReadVariableOp2z
;batch_normalization_211/AssignMovingAvg/AssignSubVariableOp;batch_normalization_211/AssignMovingAvg/AssignSubVariableOp2p
6batch_normalization_211/AssignMovingAvg/ReadVariableOp6batch_normalization_211/AssignMovingAvg/ReadVariableOp2~
=batch_normalization_211/AssignMovingAvg_1/AssignSubVariableOp=batch_normalization_211/AssignMovingAvg_1/AssignSubVariableOp2t
8batch_normalization_211/AssignMovingAvg_1/ReadVariableOp8batch_normalization_211/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_211/batchnorm/ReadVariableOp0batch_normalization_211/batchnorm/ReadVariableOp2l
4batch_normalization_211/batchnorm/mul/ReadVariableOp4batch_normalization_211/batchnorm/mul/ReadVariableOp2z
;batch_normalization_212/AssignMovingAvg/AssignSubVariableOp;batch_normalization_212/AssignMovingAvg/AssignSubVariableOp2p
6batch_normalization_212/AssignMovingAvg/ReadVariableOp6batch_normalization_212/AssignMovingAvg/ReadVariableOp2~
=batch_normalization_212/AssignMovingAvg_1/AssignSubVariableOp=batch_normalization_212/AssignMovingAvg_1/AssignSubVariableOp2t
8batch_normalization_212/AssignMovingAvg_1/ReadVariableOp8batch_normalization_212/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_212/ReadVariableOp&batch_normalization_212/ReadVariableOp2T
(batch_normalization_212/ReadVariableOp_1(batch_normalization_212/ReadVariableOp_12z
;batch_normalization_213/AssignMovingAvg/AssignSubVariableOp;batch_normalization_213/AssignMovingAvg/AssignSubVariableOp2p
6batch_normalization_213/AssignMovingAvg/ReadVariableOp6batch_normalization_213/AssignMovingAvg/ReadVariableOp2~
=batch_normalization_213/AssignMovingAvg_1/AssignSubVariableOp=batch_normalization_213/AssignMovingAvg_1/AssignSubVariableOp2t
8batch_normalization_213/AssignMovingAvg_1/ReadVariableOp8batch_normalization_213/AssignMovingAvg_1/ReadVariableOp2P
&batch_normalization_213/ReadVariableOp&batch_normalization_213/ReadVariableOp2T
(batch_normalization_213/ReadVariableOp_1(batch_normalization_213/ReadVariableOp_12j
3conv2d_transpose_78/conv2d_transpose/ReadVariableOp3conv2d_transpose_78/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_79/conv2d_transpose/ReadVariableOp3conv2d_transpose_79/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_80/conv2d_transpose/ReadVariableOp3conv2d_transpose_80/conv2d_transpose/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2D
 dense_102/BiasAdd/ReadVariableOp dense_102/BiasAdd/ReadVariableOp2B
dense_102/MatMul/ReadVariableOpdense_102/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_838592

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_838577
assignmovingavg_1_838584
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/838577*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/838577*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_838577*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/838577*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/838577*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_838577AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/838577*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/838584*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/838584*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_838584*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/838584*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/838584*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_838584AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/838584*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_838793

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
.__inference_sequential_53_layer_call_fn_839843

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_53_layer_call_and_return_conditional_losses_8392142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
I__inference_sequential_53_layer_call_and_return_conditional_losses_839787

inputs,
(dense_101_matmul_readvariableop_resource-
)dense_101_biasadd_readvariableop_resource=
9batch_normalization_210_batchnorm_readvariableop_resourceA
=batch_normalization_210_batchnorm_mul_readvariableop_resource?
;batch_normalization_210_batchnorm_readvariableop_1_resource?
;batch_normalization_210_batchnorm_readvariableop_2_resource,
(dense_102_matmul_readvariableop_resource-
)dense_102_biasadd_readvariableop_resource=
9batch_normalization_211_batchnorm_readvariableop_resourceA
=batch_normalization_211_batchnorm_mul_readvariableop_resource?
;batch_normalization_211_batchnorm_readvariableop_1_resource?
;batch_normalization_211_batchnorm_readvariableop_2_resource@
<conv2d_transpose_78_conv2d_transpose_readvariableop_resource3
/batch_normalization_212_readvariableop_resource5
1batch_normalization_212_readvariableop_1_resourceD
@batch_normalization_212_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_212_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_79_conv2d_transpose_readvariableop_resource3
/batch_normalization_213_readvariableop_resource5
1batch_normalization_213_readvariableop_1_resourceD
@batch_normalization_213_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_213_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_80_conv2d_transpose_readvariableop_resource
identity��0batch_normalization_210/batchnorm/ReadVariableOp�2batch_normalization_210/batchnorm/ReadVariableOp_1�2batch_normalization_210/batchnorm/ReadVariableOp_2�4batch_normalization_210/batchnorm/mul/ReadVariableOp�0batch_normalization_211/batchnorm/ReadVariableOp�2batch_normalization_211/batchnorm/ReadVariableOp_1�2batch_normalization_211/batchnorm/ReadVariableOp_2�4batch_normalization_211/batchnorm/mul/ReadVariableOp�7batch_normalization_212/FusedBatchNormV3/ReadVariableOp�9batch_normalization_212/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_212/ReadVariableOp�(batch_normalization_212/ReadVariableOp_1�7batch_normalization_213/FusedBatchNormV3/ReadVariableOp�9batch_normalization_213/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_213/ReadVariableOp�(batch_normalization_213/ReadVariableOp_1�3conv2d_transpose_78/conv2d_transpose/ReadVariableOp�3conv2d_transpose_79/conv2d_transpose/ReadVariableOp�3conv2d_transpose_80/conv2d_transpose/ReadVariableOp� dense_101/BiasAdd/ReadVariableOp�dense_101/MatMul/ReadVariableOp� dense_102/BiasAdd/ReadVariableOp�dense_102/MatMul/ReadVariableOp�
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02!
dense_101/MatMul/ReadVariableOp�
dense_101/MatMulMatMulinputs'dense_101/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_101/MatMul�
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dense_101/BiasAdd/ReadVariableOp�
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_101/BiasAdd�
$batch_normalization_210/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2&
$batch_normalization_210/LogicalAnd/x�
$batch_normalization_210/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_210/LogicalAnd/y�
"batch_normalization_210/LogicalAnd
LogicalAnd-batch_normalization_210/LogicalAnd/x:output:0-batch_normalization_210/LogicalAnd/y:output:0*
_output_shapes
: 2$
"batch_normalization_210/LogicalAnd�
0batch_normalization_210/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_210_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_210/batchnorm/ReadVariableOp�
'batch_normalization_210/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2)
'batch_normalization_210/batchnorm/add/y�
%batch_normalization_210/batchnorm/addAddV28batch_normalization_210/batchnorm/ReadVariableOp:value:00batch_normalization_210/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2'
%batch_normalization_210/batchnorm/add�
'batch_normalization_210/batchnorm/RsqrtRsqrt)batch_normalization_210/batchnorm/add:z:0*
T0*
_output_shapes	
:�2)
'batch_normalization_210/batchnorm/Rsqrt�
4batch_normalization_210/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_210_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_210/batchnorm/mul/ReadVariableOp�
%batch_normalization_210/batchnorm/mulMul+batch_normalization_210/batchnorm/Rsqrt:y:0<batch_normalization_210/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2'
%batch_normalization_210/batchnorm/mul�
'batch_normalization_210/batchnorm/mul_1Muldense_101/BiasAdd:output:0)batch_normalization_210/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2)
'batch_normalization_210/batchnorm/mul_1�
2batch_normalization_210/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_210_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_210/batchnorm/ReadVariableOp_1�
'batch_normalization_210/batchnorm/mul_2Mul:batch_normalization_210/batchnorm/ReadVariableOp_1:value:0)batch_normalization_210/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2)
'batch_normalization_210/batchnorm/mul_2�
2batch_normalization_210/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_210_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_210/batchnorm/ReadVariableOp_2�
%batch_normalization_210/batchnorm/subSub:batch_normalization_210/batchnorm/ReadVariableOp_2:value:0+batch_normalization_210/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_210/batchnorm/sub�
'batch_normalization_210/batchnorm/add_1AddV2+batch_normalization_210/batchnorm/mul_1:z:0)batch_normalization_210/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2)
'batch_normalization_210/batchnorm/add_1�
leaky_re_lu_207/LeakyRelu	LeakyRelu+batch_normalization_210/batchnorm/add_1:z:0*(
_output_shapes
:����������2
leaky_re_lu_207/LeakyRelu�
dense_102/MatMul/ReadVariableOpReadVariableOp(dense_102_matmul_readvariableop_resource* 
_output_shapes
:
��b*
dtype02!
dense_102/MatMul/ReadVariableOp�
dense_102/MatMulMatMul'leaky_re_lu_207/LeakyRelu:activations:0'dense_102/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b2
dense_102/MatMul�
 dense_102/BiasAdd/ReadVariableOpReadVariableOp)dense_102_biasadd_readvariableop_resource*
_output_shapes	
:�b*
dtype02"
 dense_102/BiasAdd/ReadVariableOp�
dense_102/BiasAddBiasAdddense_102/MatMul:product:0(dense_102/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b2
dense_102/BiasAdd�
$batch_normalization_211/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2&
$batch_normalization_211/LogicalAnd/x�
$batch_normalization_211/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_211/LogicalAnd/y�
"batch_normalization_211/LogicalAnd
LogicalAnd-batch_normalization_211/LogicalAnd/x:output:0-batch_normalization_211/LogicalAnd/y:output:0*
_output_shapes
: 2$
"batch_normalization_211/LogicalAnd�
0batch_normalization_211/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_211_batchnorm_readvariableop_resource*
_output_shapes	
:�b*
dtype022
0batch_normalization_211/batchnorm/ReadVariableOp�
'batch_normalization_211/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2)
'batch_normalization_211/batchnorm/add/y�
%batch_normalization_211/batchnorm/addAddV28batch_normalization_211/batchnorm/ReadVariableOp:value:00batch_normalization_211/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�b2'
%batch_normalization_211/batchnorm/add�
'batch_normalization_211/batchnorm/RsqrtRsqrt)batch_normalization_211/batchnorm/add:z:0*
T0*
_output_shapes	
:�b2)
'batch_normalization_211/batchnorm/Rsqrt�
4batch_normalization_211/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_211_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�b*
dtype026
4batch_normalization_211/batchnorm/mul/ReadVariableOp�
%batch_normalization_211/batchnorm/mulMul+batch_normalization_211/batchnorm/Rsqrt:y:0<batch_normalization_211/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b2'
%batch_normalization_211/batchnorm/mul�
'batch_normalization_211/batchnorm/mul_1Muldense_102/BiasAdd:output:0)batch_normalization_211/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������b2)
'batch_normalization_211/batchnorm/mul_1�
2batch_normalization_211/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_211_batchnorm_readvariableop_1_resource*
_output_shapes	
:�b*
dtype024
2batch_normalization_211/batchnorm/ReadVariableOp_1�
'batch_normalization_211/batchnorm/mul_2Mul:batch_normalization_211/batchnorm/ReadVariableOp_1:value:0)batch_normalization_211/batchnorm/mul:z:0*
T0*
_output_shapes	
:�b2)
'batch_normalization_211/batchnorm/mul_2�
2batch_normalization_211/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_211_batchnorm_readvariableop_2_resource*
_output_shapes	
:�b*
dtype024
2batch_normalization_211/batchnorm/ReadVariableOp_2�
%batch_normalization_211/batchnorm/subSub:batch_normalization_211/batchnorm/ReadVariableOp_2:value:0+batch_normalization_211/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�b2'
%batch_normalization_211/batchnorm/sub�
'batch_normalization_211/batchnorm/add_1AddV2+batch_normalization_211/batchnorm/mul_1:z:0)batch_normalization_211/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������b2)
'batch_normalization_211/batchnorm/add_1�
leaky_re_lu_208/LeakyRelu	LeakyRelu+batch_normalization_211/batchnorm/add_1:z:0*(
_output_shapes
:����������b2
leaky_re_lu_208/LeakyRelu{
reshape_26/ShapeShape'leaky_re_lu_208/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_26/Shape�
reshape_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_26/strided_slice/stack�
 reshape_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_26/strided_slice/stack_1�
 reshape_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_26/strided_slice/stack_2�
reshape_26/strided_sliceStridedSlicereshape_26/Shape:output:0'reshape_26/strided_slice/stack:output:0)reshape_26/strided_slice/stack_1:output:0)reshape_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_26/strided_slicez
reshape_26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_26/Reshape/shape/1z
reshape_26/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_26/Reshape/shape/2{
reshape_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :�2
reshape_26/Reshape/shape/3�
reshape_26/Reshape/shapePack!reshape_26/strided_slice:output:0#reshape_26/Reshape/shape/1:output:0#reshape_26/Reshape/shape/2:output:0#reshape_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_26/Reshape/shape�
reshape_26/ReshapeReshape'leaky_re_lu_208/LeakyRelu:activations:0!reshape_26/Reshape/shape:output:0*
T0*0
_output_shapes
:����������2
reshape_26/Reshape�
conv2d_transpose_78/ShapeShapereshape_26/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_78/Shape�
'conv2d_transpose_78/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_78/strided_slice/stack�
)conv2d_transpose_78/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_78/strided_slice/stack_1�
)conv2d_transpose_78/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_78/strided_slice/stack_2�
!conv2d_transpose_78/strided_sliceStridedSlice"conv2d_transpose_78/Shape:output:00conv2d_transpose_78/strided_slice/stack:output:02conv2d_transpose_78/strided_slice/stack_1:output:02conv2d_transpose_78/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_78/strided_slice�
)conv2d_transpose_78/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_78/strided_slice_1/stack�
+conv2d_transpose_78/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_1/stack_1�
+conv2d_transpose_78/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_1/stack_2�
#conv2d_transpose_78/strided_slice_1StridedSlice"conv2d_transpose_78/Shape:output:02conv2d_transpose_78/strided_slice_1/stack:output:04conv2d_transpose_78/strided_slice_1/stack_1:output:04conv2d_transpose_78/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_78/strided_slice_1�
)conv2d_transpose_78/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_78/strided_slice_2/stack�
+conv2d_transpose_78/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_2/stack_1�
+conv2d_transpose_78/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_2/stack_2�
#conv2d_transpose_78/strided_slice_2StridedSlice"conv2d_transpose_78/Shape:output:02conv2d_transpose_78/strided_slice_2/stack:output:04conv2d_transpose_78/strided_slice_2/stack_1:output:04conv2d_transpose_78/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_78/strided_slice_2x
conv2d_transpose_78/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_78/mul/y�
conv2d_transpose_78/mulMul,conv2d_transpose_78/strided_slice_1:output:0"conv2d_transpose_78/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_78/mul|
conv2d_transpose_78/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_78/mul_1/y�
conv2d_transpose_78/mul_1Mul,conv2d_transpose_78/strided_slice_2:output:0$conv2d_transpose_78/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_78/mul_1}
conv2d_transpose_78/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
conv2d_transpose_78/stack/3�
conv2d_transpose_78/stackPack*conv2d_transpose_78/strided_slice:output:0conv2d_transpose_78/mul:z:0conv2d_transpose_78/mul_1:z:0$conv2d_transpose_78/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_78/stack�
)conv2d_transpose_78/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_78/strided_slice_3/stack�
+conv2d_transpose_78/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_3/stack_1�
+conv2d_transpose_78/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_78/strided_slice_3/stack_2�
#conv2d_transpose_78/strided_slice_3StridedSlice"conv2d_transpose_78/stack:output:02conv2d_transpose_78/strided_slice_3/stack:output:04conv2d_transpose_78/strided_slice_3/stack_1:output:04conv2d_transpose_78/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_78/strided_slice_3�
3conv2d_transpose_78/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_78_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype025
3conv2d_transpose_78/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_78/conv2d_transposeConv2DBackpropInput"conv2d_transpose_78/stack:output:0;conv2d_transpose_78/conv2d_transpose/ReadVariableOp:value:0reshape_26/Reshape:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$conv2d_transpose_78/conv2d_transpose�
$batch_normalization_212/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2&
$batch_normalization_212/LogicalAnd/x�
$batch_normalization_212/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_212/LogicalAnd/y�
"batch_normalization_212/LogicalAnd
LogicalAnd-batch_normalization_212/LogicalAnd/x:output:0-batch_normalization_212/LogicalAnd/y:output:0*
_output_shapes
: 2$
"batch_normalization_212/LogicalAnd�
&batch_normalization_212/ReadVariableOpReadVariableOp/batch_normalization_212_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_212/ReadVariableOp�
(batch_normalization_212/ReadVariableOp_1ReadVariableOp1batch_normalization_212_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_212/ReadVariableOp_1�
7batch_normalization_212/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_212_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_212/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_212/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_212_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_212/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_212/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_78/conv2d_transpose:output:0.batch_normalization_212/ReadVariableOp:value:00batch_normalization_212/ReadVariableOp_1:value:0?batch_normalization_212/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_212/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2*
(batch_normalization_212/FusedBatchNormV3�
batch_normalization_212/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_212/Const�
leaky_re_lu_209/LeakyRelu	LeakyRelu,batch_normalization_212/FusedBatchNormV3:y:0*0
_output_shapes
:����������2
leaky_re_lu_209/LeakyRelu�
conv2d_transpose_79/ShapeShape'leaky_re_lu_209/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_79/Shape�
'conv2d_transpose_79/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_79/strided_slice/stack�
)conv2d_transpose_79/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_79/strided_slice/stack_1�
)conv2d_transpose_79/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_79/strided_slice/stack_2�
!conv2d_transpose_79/strided_sliceStridedSlice"conv2d_transpose_79/Shape:output:00conv2d_transpose_79/strided_slice/stack:output:02conv2d_transpose_79/strided_slice/stack_1:output:02conv2d_transpose_79/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_79/strided_slice�
)conv2d_transpose_79/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_79/strided_slice_1/stack�
+conv2d_transpose_79/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_1/stack_1�
+conv2d_transpose_79/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_1/stack_2�
#conv2d_transpose_79/strided_slice_1StridedSlice"conv2d_transpose_79/Shape:output:02conv2d_transpose_79/strided_slice_1/stack:output:04conv2d_transpose_79/strided_slice_1/stack_1:output:04conv2d_transpose_79/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_79/strided_slice_1�
)conv2d_transpose_79/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_79/strided_slice_2/stack�
+conv2d_transpose_79/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_2/stack_1�
+conv2d_transpose_79/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_2/stack_2�
#conv2d_transpose_79/strided_slice_2StridedSlice"conv2d_transpose_79/Shape:output:02conv2d_transpose_79/strided_slice_2/stack:output:04conv2d_transpose_79/strided_slice_2/stack_1:output:04conv2d_transpose_79/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_79/strided_slice_2x
conv2d_transpose_79/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_79/mul/y�
conv2d_transpose_79/mulMul,conv2d_transpose_79/strided_slice_1:output:0"conv2d_transpose_79/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_79/mul|
conv2d_transpose_79/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_79/mul_1/y�
conv2d_transpose_79/mul_1Mul,conv2d_transpose_79/strided_slice_2:output:0$conv2d_transpose_79/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_79/mul_1|
conv2d_transpose_79/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_79/stack/3�
conv2d_transpose_79/stackPack*conv2d_transpose_79/strided_slice:output:0conv2d_transpose_79/mul:z:0conv2d_transpose_79/mul_1:z:0$conv2d_transpose_79/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_79/stack�
)conv2d_transpose_79/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_79/strided_slice_3/stack�
+conv2d_transpose_79/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_3/stack_1�
+conv2d_transpose_79/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_79/strided_slice_3/stack_2�
#conv2d_transpose_79/strided_slice_3StridedSlice"conv2d_transpose_79/stack:output:02conv2d_transpose_79/strided_slice_3/stack:output:04conv2d_transpose_79/strided_slice_3/stack_1:output:04conv2d_transpose_79/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_79/strided_slice_3�
3conv2d_transpose_79/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_79_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype025
3conv2d_transpose_79/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_79/conv2d_transposeConv2DBackpropInput"conv2d_transpose_79/stack:output:0;conv2d_transpose_79/conv2d_transpose/ReadVariableOp:value:0'leaky_re_lu_209/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2&
$conv2d_transpose_79/conv2d_transpose�
$batch_normalization_213/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2&
$batch_normalization_213/LogicalAnd/x�
$batch_normalization_213/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2&
$batch_normalization_213/LogicalAnd/y�
"batch_normalization_213/LogicalAnd
LogicalAnd-batch_normalization_213/LogicalAnd/x:output:0-batch_normalization_213/LogicalAnd/y:output:0*
_output_shapes
: 2$
"batch_normalization_213/LogicalAnd�
&batch_normalization_213/ReadVariableOpReadVariableOp/batch_normalization_213_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_213/ReadVariableOp�
(batch_normalization_213/ReadVariableOp_1ReadVariableOp1batch_normalization_213_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_213/ReadVariableOp_1�
7batch_normalization_213/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_213_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_213/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_213/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_213_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_213/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_213/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_79/conv2d_transpose:output:0.batch_normalization_213/ReadVariableOp:value:00batch_normalization_213/ReadVariableOp_1:value:0?batch_normalization_213/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_213/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2*
(batch_normalization_213/FusedBatchNormV3�
batch_normalization_213/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_213/Const�
leaky_re_lu_210/LeakyRelu	LeakyRelu,batch_normalization_213/FusedBatchNormV3:y:0*/
_output_shapes
:���������@2
leaky_re_lu_210/LeakyRelu�
conv2d_transpose_80/ShapeShape'leaky_re_lu_210/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_80/Shape�
'conv2d_transpose_80/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_80/strided_slice/stack�
)conv2d_transpose_80/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice/stack_1�
)conv2d_transpose_80/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice/stack_2�
!conv2d_transpose_80/strided_sliceStridedSlice"conv2d_transpose_80/Shape:output:00conv2d_transpose_80/strided_slice/stack:output:02conv2d_transpose_80/strided_slice/stack_1:output:02conv2d_transpose_80/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_80/strided_slice�
)conv2d_transpose_80/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice_1/stack�
+conv2d_transpose_80/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_1/stack_1�
+conv2d_transpose_80/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_1/stack_2�
#conv2d_transpose_80/strided_slice_1StridedSlice"conv2d_transpose_80/Shape:output:02conv2d_transpose_80/strided_slice_1/stack:output:04conv2d_transpose_80/strided_slice_1/stack_1:output:04conv2d_transpose_80/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_80/strided_slice_1�
)conv2d_transpose_80/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice_2/stack�
+conv2d_transpose_80/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_2/stack_1�
+conv2d_transpose_80/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_2/stack_2�
#conv2d_transpose_80/strided_slice_2StridedSlice"conv2d_transpose_80/Shape:output:02conv2d_transpose_80/strided_slice_2/stack:output:04conv2d_transpose_80/strided_slice_2/stack_1:output:04conv2d_transpose_80/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_80/strided_slice_2x
conv2d_transpose_80/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_80/mul/y�
conv2d_transpose_80/mulMul,conv2d_transpose_80/strided_slice_1:output:0"conv2d_transpose_80/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_80/mul|
conv2d_transpose_80/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_80/mul_1/y�
conv2d_transpose_80/mul_1Mul,conv2d_transpose_80/strided_slice_2:output:0$conv2d_transpose_80/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_80/mul_1|
conv2d_transpose_80/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_80/stack/3�
conv2d_transpose_80/stackPack*conv2d_transpose_80/strided_slice:output:0conv2d_transpose_80/mul:z:0conv2d_transpose_80/mul_1:z:0$conv2d_transpose_80/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_80/stack�
)conv2d_transpose_80/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_80/strided_slice_3/stack�
+conv2d_transpose_80/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_3/stack_1�
+conv2d_transpose_80/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_3/stack_2�
#conv2d_transpose_80/strided_slice_3StridedSlice"conv2d_transpose_80/stack:output:02conv2d_transpose_80/strided_slice_3/stack:output:04conv2d_transpose_80/strided_slice_3/stack_1:output:04conv2d_transpose_80/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_80/strided_slice_3�
3conv2d_transpose_80/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_80_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype025
3conv2d_transpose_80/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_80/conv2d_transposeConv2DBackpropInput"conv2d_transpose_80/stack:output:0;conv2d_transpose_80/conv2d_transpose/ReadVariableOp:value:0'leaky_re_lu_210/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2&
$conv2d_transpose_80/conv2d_transpose�
conv2d_transpose_80/TanhTanh-conv2d_transpose_80/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������2
conv2d_transpose_80/Tanh�	
IdentityIdentityconv2d_transpose_80/Tanh:y:01^batch_normalization_210/batchnorm/ReadVariableOp3^batch_normalization_210/batchnorm/ReadVariableOp_13^batch_normalization_210/batchnorm/ReadVariableOp_25^batch_normalization_210/batchnorm/mul/ReadVariableOp1^batch_normalization_211/batchnorm/ReadVariableOp3^batch_normalization_211/batchnorm/ReadVariableOp_13^batch_normalization_211/batchnorm/ReadVariableOp_25^batch_normalization_211/batchnorm/mul/ReadVariableOp8^batch_normalization_212/FusedBatchNormV3/ReadVariableOp:^batch_normalization_212/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_212/ReadVariableOp)^batch_normalization_212/ReadVariableOp_18^batch_normalization_213/FusedBatchNormV3/ReadVariableOp:^batch_normalization_213/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_213/ReadVariableOp)^batch_normalization_213/ReadVariableOp_14^conv2d_transpose_78/conv2d_transpose/ReadVariableOp4^conv2d_transpose_79/conv2d_transpose/ReadVariableOp4^conv2d_transpose_80/conv2d_transpose/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp!^dense_102/BiasAdd/ReadVariableOp ^dense_102/MatMul/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::2d
0batch_normalization_210/batchnorm/ReadVariableOp0batch_normalization_210/batchnorm/ReadVariableOp2h
2batch_normalization_210/batchnorm/ReadVariableOp_12batch_normalization_210/batchnorm/ReadVariableOp_12h
2batch_normalization_210/batchnorm/ReadVariableOp_22batch_normalization_210/batchnorm/ReadVariableOp_22l
4batch_normalization_210/batchnorm/mul/ReadVariableOp4batch_normalization_210/batchnorm/mul/ReadVariableOp2d
0batch_normalization_211/batchnorm/ReadVariableOp0batch_normalization_211/batchnorm/ReadVariableOp2h
2batch_normalization_211/batchnorm/ReadVariableOp_12batch_normalization_211/batchnorm/ReadVariableOp_12h
2batch_normalization_211/batchnorm/ReadVariableOp_22batch_normalization_211/batchnorm/ReadVariableOp_22l
4batch_normalization_211/batchnorm/mul/ReadVariableOp4batch_normalization_211/batchnorm/mul/ReadVariableOp2r
7batch_normalization_212/FusedBatchNormV3/ReadVariableOp7batch_normalization_212/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_212/FusedBatchNormV3/ReadVariableOp_19batch_normalization_212/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_212/ReadVariableOp&batch_normalization_212/ReadVariableOp2T
(batch_normalization_212/ReadVariableOp_1(batch_normalization_212/ReadVariableOp_12r
7batch_normalization_213/FusedBatchNormV3/ReadVariableOp7batch_normalization_213/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_213/FusedBatchNormV3/ReadVariableOp_19batch_normalization_213/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_213/ReadVariableOp&batch_normalization_213/ReadVariableOp2T
(batch_normalization_213/ReadVariableOp_1(batch_normalization_213/ReadVariableOp_12j
3conv2d_transpose_78/conv2d_transpose/ReadVariableOp3conv2d_transpose_78/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_79/conv2d_transpose/ReadVariableOp3conv2d_transpose_79/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_80/conv2d_transpose/ReadVariableOp3conv2d_transpose_80/conv2d_transpose/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2D
 dense_102/BiasAdd/ReadVariableOp dense_102/BiasAdd/ReadVariableOp2B
dense_102/MatMul/ReadVariableOpdense_102/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�R
�
I__inference_sequential_53_layer_call_and_return_conditional_losses_839060
dense_101_input,
(dense_101_statefulpartitionedcall_args_1,
(dense_101_statefulpartitionedcall_args_2:
6batch_normalization_210_statefulpartitionedcall_args_1:
6batch_normalization_210_statefulpartitionedcall_args_2:
6batch_normalization_210_statefulpartitionedcall_args_3:
6batch_normalization_210_statefulpartitionedcall_args_4,
(dense_102_statefulpartitionedcall_args_1,
(dense_102_statefulpartitionedcall_args_2:
6batch_normalization_211_statefulpartitionedcall_args_1:
6batch_normalization_211_statefulpartitionedcall_args_2:
6batch_normalization_211_statefulpartitionedcall_args_3:
6batch_normalization_211_statefulpartitionedcall_args_46
2conv2d_transpose_78_statefulpartitionedcall_args_1:
6batch_normalization_212_statefulpartitionedcall_args_1:
6batch_normalization_212_statefulpartitionedcall_args_2:
6batch_normalization_212_statefulpartitionedcall_args_3:
6batch_normalization_212_statefulpartitionedcall_args_46
2conv2d_transpose_79_statefulpartitionedcall_args_1:
6batch_normalization_213_statefulpartitionedcall_args_1:
6batch_normalization_213_statefulpartitionedcall_args_2:
6batch_normalization_213_statefulpartitionedcall_args_3:
6batch_normalization_213_statefulpartitionedcall_args_46
2conv2d_transpose_80_statefulpartitionedcall_args_1
identity��/batch_normalization_210/StatefulPartitionedCall�/batch_normalization_211/StatefulPartitionedCall�/batch_normalization_212/StatefulPartitionedCall�/batch_normalization_213/StatefulPartitionedCall�+conv2d_transpose_78/StatefulPartitionedCall�+conv2d_transpose_79/StatefulPartitionedCall�+conv2d_transpose_80/StatefulPartitionedCall�!dense_101/StatefulPartitionedCall�!dense_102/StatefulPartitionedCall�
!dense_101/StatefulPartitionedCallStatefulPartitionedCalldense_101_input(dense_101_statefulpartitionedcall_args_1(dense_101_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_101_layer_call_and_return_conditional_losses_8388532#
!dense_101/StatefulPartitionedCall�
/batch_normalization_210/StatefulPartitionedCallStatefulPartitionedCall*dense_101/StatefulPartitionedCall:output:06batch_normalization_210_statefulpartitionedcall_args_16batch_normalization_210_statefulpartitionedcall_args_26batch_normalization_210_statefulpartitionedcall_args_36batch_normalization_210_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_83827721
/batch_normalization_210/StatefulPartitionedCall�
leaky_re_lu_207/PartitionedCallPartitionedCall8batch_normalization_210/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_207_layer_call_and_return_conditional_losses_8388932!
leaky_re_lu_207/PartitionedCall�
!dense_102/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_207/PartitionedCall:output:0(dense_102_statefulpartitionedcall_args_1(dense_102_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_102_layer_call_and_return_conditional_losses_8389112#
!dense_102/StatefulPartitionedCall�
/batch_normalization_211/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:06batch_normalization_211_statefulpartitionedcall_args_16batch_normalization_211_statefulpartitionedcall_args_26batch_normalization_211_statefulpartitionedcall_args_36batch_normalization_211_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_83842121
/batch_normalization_211/StatefulPartitionedCall�
leaky_re_lu_208/PartitionedCallPartitionedCall8batch_normalization_211/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_208_layer_call_and_return_conditional_losses_8389512!
leaky_re_lu_208/PartitionedCall�
reshape_26/PartitionedCallPartitionedCall(leaky_re_lu_208/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_reshape_26_layer_call_and_return_conditional_losses_8389732
reshape_26/PartitionedCall�
+conv2d_transpose_78/StatefulPartitionedCallStatefulPartitionedCall#reshape_26/PartitionedCall:output:02conv2d_transpose_78_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_78_layer_call_and_return_conditional_losses_8384912-
+conv2d_transpose_78/StatefulPartitionedCall�
/batch_normalization_212/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_78/StatefulPartitionedCall:output:06batch_normalization_212_statefulpartitionedcall_args_16batch_normalization_212_statefulpartitionedcall_args_26batch_normalization_212_statefulpartitionedcall_args_36batch_normalization_212_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_83859221
/batch_normalization_212/StatefulPartitionedCall�
leaky_re_lu_209/PartitionedCallPartitionedCall8batch_normalization_212/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_209_layer_call_and_return_conditional_losses_8390112!
leaky_re_lu_209/PartitionedCall�
+conv2d_transpose_79/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_209/PartitionedCall:output:02conv2d_transpose_79_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_79_layer_call_and_return_conditional_losses_8386612-
+conv2d_transpose_79/StatefulPartitionedCall�
/batch_normalization_213/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_79/StatefulPartitionedCall:output:06batch_normalization_213_statefulpartitionedcall_args_16batch_normalization_213_statefulpartitionedcall_args_26batch_normalization_213_statefulpartitionedcall_args_36batch_normalization_213_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_83876221
/batch_normalization_213/StatefulPartitionedCall�
leaky_re_lu_210/PartitionedCallPartitionedCall8batch_normalization_213/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_210_layer_call_and_return_conditional_losses_8390492!
leaky_re_lu_210/PartitionedCall�
+conv2d_transpose_80/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_210/PartitionedCall:output:02conv2d_transpose_80_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_8388322-
+conv2d_transpose_80/StatefulPartitionedCall�
IdentityIdentity4conv2d_transpose_80/StatefulPartitionedCall:output:00^batch_normalization_210/StatefulPartitionedCall0^batch_normalization_211/StatefulPartitionedCall0^batch_normalization_212/StatefulPartitionedCall0^batch_normalization_213/StatefulPartitionedCall,^conv2d_transpose_78/StatefulPartitionedCall,^conv2d_transpose_79/StatefulPartitionedCall,^conv2d_transpose_80/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::2b
/batch_normalization_210/StatefulPartitionedCall/batch_normalization_210/StatefulPartitionedCall2b
/batch_normalization_211/StatefulPartitionedCall/batch_normalization_211/StatefulPartitionedCall2b
/batch_normalization_212/StatefulPartitionedCall/batch_normalization_212/StatefulPartitionedCall2b
/batch_normalization_213/StatefulPartitionedCall/batch_normalization_213/StatefulPartitionedCall2Z
+conv2d_transpose_78/StatefulPartitionedCall+conv2d_transpose_78/StatefulPartitionedCall2Z
+conv2d_transpose_79/StatefulPartitionedCall+conv2d_transpose_79/StatefulPartitionedCall2Z
+conv2d_transpose_80/StatefulPartitionedCall+conv2d_transpose_80/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall:/ +
)
_user_specified_namedense_101_input
�
b
F__inference_reshape_26_layer_call_and_return_conditional_losses_840143

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :�2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:����������2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������b:& "
 
_user_specified_nameinputs
�
�
.__inference_sequential_53_layer_call_fn_839815

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_53_layer_call_and_return_conditional_losses_8391452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_212_layer_call_fn_840225

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_8385922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
4__inference_conv2d_transpose_79_layer_call_fn_838668

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_79_layer_call_and_return_conditional_losses_8386612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,����������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_213_layer_call_fn_840321

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_8387622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
L
0__inference_leaky_re_lu_208_layer_call_fn_840129

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_208_layer_call_and_return_conditional_losses_8389512
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������b:& "
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_838172
dense_101_input:
6sequential_53_dense_101_matmul_readvariableop_resource;
7sequential_53_dense_101_biasadd_readvariableop_resourceK
Gsequential_53_batch_normalization_210_batchnorm_readvariableop_resourceO
Ksequential_53_batch_normalization_210_batchnorm_mul_readvariableop_resourceM
Isequential_53_batch_normalization_210_batchnorm_readvariableop_1_resourceM
Isequential_53_batch_normalization_210_batchnorm_readvariableop_2_resource:
6sequential_53_dense_102_matmul_readvariableop_resource;
7sequential_53_dense_102_biasadd_readvariableop_resourceK
Gsequential_53_batch_normalization_211_batchnorm_readvariableop_resourceO
Ksequential_53_batch_normalization_211_batchnorm_mul_readvariableop_resourceM
Isequential_53_batch_normalization_211_batchnorm_readvariableop_1_resourceM
Isequential_53_batch_normalization_211_batchnorm_readvariableop_2_resourceN
Jsequential_53_conv2d_transpose_78_conv2d_transpose_readvariableop_resourceA
=sequential_53_batch_normalization_212_readvariableop_resourceC
?sequential_53_batch_normalization_212_readvariableop_1_resourceR
Nsequential_53_batch_normalization_212_fusedbatchnormv3_readvariableop_resourceT
Psequential_53_batch_normalization_212_fusedbatchnormv3_readvariableop_1_resourceN
Jsequential_53_conv2d_transpose_79_conv2d_transpose_readvariableop_resourceA
=sequential_53_batch_normalization_213_readvariableop_resourceC
?sequential_53_batch_normalization_213_readvariableop_1_resourceR
Nsequential_53_batch_normalization_213_fusedbatchnormv3_readvariableop_resourceT
Psequential_53_batch_normalization_213_fusedbatchnormv3_readvariableop_1_resourceN
Jsequential_53_conv2d_transpose_80_conv2d_transpose_readvariableop_resource
identity��>sequential_53/batch_normalization_210/batchnorm/ReadVariableOp�@sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_1�@sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_2�Bsequential_53/batch_normalization_210/batchnorm/mul/ReadVariableOp�>sequential_53/batch_normalization_211/batchnorm/ReadVariableOp�@sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_1�@sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_2�Bsequential_53/batch_normalization_211/batchnorm/mul/ReadVariableOp�Esequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOp�Gsequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOp_1�4sequential_53/batch_normalization_212/ReadVariableOp�6sequential_53/batch_normalization_212/ReadVariableOp_1�Esequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOp�Gsequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOp_1�4sequential_53/batch_normalization_213/ReadVariableOp�6sequential_53/batch_normalization_213/ReadVariableOp_1�Asequential_53/conv2d_transpose_78/conv2d_transpose/ReadVariableOp�Asequential_53/conv2d_transpose_79/conv2d_transpose/ReadVariableOp�Asequential_53/conv2d_transpose_80/conv2d_transpose/ReadVariableOp�.sequential_53/dense_101/BiasAdd/ReadVariableOp�-sequential_53/dense_101/MatMul/ReadVariableOp�.sequential_53/dense_102/BiasAdd/ReadVariableOp�-sequential_53/dense_102/MatMul/ReadVariableOp�
-sequential_53/dense_101/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_101_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02/
-sequential_53/dense_101/MatMul/ReadVariableOp�
sequential_53/dense_101/MatMulMatMuldense_101_input5sequential_53/dense_101/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_53/dense_101/MatMul�
.sequential_53/dense_101/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_101_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.sequential_53/dense_101/BiasAdd/ReadVariableOp�
sequential_53/dense_101/BiasAddBiasAdd(sequential_53/dense_101/MatMul:product:06sequential_53/dense_101/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2!
sequential_53/dense_101/BiasAdd�
2sequential_53/batch_normalization_210/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 24
2sequential_53/batch_normalization_210/LogicalAnd/x�
2sequential_53/batch_normalization_210/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z24
2sequential_53/batch_normalization_210/LogicalAnd/y�
0sequential_53/batch_normalization_210/LogicalAnd
LogicalAnd;sequential_53/batch_normalization_210/LogicalAnd/x:output:0;sequential_53/batch_normalization_210/LogicalAnd/y:output:0*
_output_shapes
: 22
0sequential_53/batch_normalization_210/LogicalAnd�
>sequential_53/batch_normalization_210/batchnorm/ReadVariableOpReadVariableOpGsequential_53_batch_normalization_210_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>sequential_53/batch_normalization_210/batchnorm/ReadVariableOp�
5sequential_53/batch_normalization_210/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:27
5sequential_53/batch_normalization_210/batchnorm/add/y�
3sequential_53/batch_normalization_210/batchnorm/addAddV2Fsequential_53/batch_normalization_210/batchnorm/ReadVariableOp:value:0>sequential_53/batch_normalization_210/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�25
3sequential_53/batch_normalization_210/batchnorm/add�
5sequential_53/batch_normalization_210/batchnorm/RsqrtRsqrt7sequential_53/batch_normalization_210/batchnorm/add:z:0*
T0*
_output_shapes	
:�27
5sequential_53/batch_normalization_210/batchnorm/Rsqrt�
Bsequential_53/batch_normalization_210/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_53_batch_normalization_210_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02D
Bsequential_53/batch_normalization_210/batchnorm/mul/ReadVariableOp�
3sequential_53/batch_normalization_210/batchnorm/mulMul9sequential_53/batch_normalization_210/batchnorm/Rsqrt:y:0Jsequential_53/batch_normalization_210/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�25
3sequential_53/batch_normalization_210/batchnorm/mul�
5sequential_53/batch_normalization_210/batchnorm/mul_1Mul(sequential_53/dense_101/BiasAdd:output:07sequential_53/batch_normalization_210/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������27
5sequential_53/batch_normalization_210/batchnorm/mul_1�
@sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_53_batch_normalization_210_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02B
@sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_1�
5sequential_53/batch_normalization_210/batchnorm/mul_2MulHsequential_53/batch_normalization_210/batchnorm/ReadVariableOp_1:value:07sequential_53/batch_normalization_210/batchnorm/mul:z:0*
T0*
_output_shapes	
:�27
5sequential_53/batch_normalization_210/batchnorm/mul_2�
@sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_53_batch_normalization_210_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02B
@sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_2�
3sequential_53/batch_normalization_210/batchnorm/subSubHsequential_53/batch_normalization_210/batchnorm/ReadVariableOp_2:value:09sequential_53/batch_normalization_210/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�25
3sequential_53/batch_normalization_210/batchnorm/sub�
5sequential_53/batch_normalization_210/batchnorm/add_1AddV29sequential_53/batch_normalization_210/batchnorm/mul_1:z:07sequential_53/batch_normalization_210/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������27
5sequential_53/batch_normalization_210/batchnorm/add_1�
'sequential_53/leaky_re_lu_207/LeakyRelu	LeakyRelu9sequential_53/batch_normalization_210/batchnorm/add_1:z:0*(
_output_shapes
:����������2)
'sequential_53/leaky_re_lu_207/LeakyRelu�
-sequential_53/dense_102/MatMul/ReadVariableOpReadVariableOp6sequential_53_dense_102_matmul_readvariableop_resource* 
_output_shapes
:
��b*
dtype02/
-sequential_53/dense_102/MatMul/ReadVariableOp�
sequential_53/dense_102/MatMulMatMul5sequential_53/leaky_re_lu_207/LeakyRelu:activations:05sequential_53/dense_102/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b2 
sequential_53/dense_102/MatMul�
.sequential_53/dense_102/BiasAdd/ReadVariableOpReadVariableOp7sequential_53_dense_102_biasadd_readvariableop_resource*
_output_shapes	
:�b*
dtype020
.sequential_53/dense_102/BiasAdd/ReadVariableOp�
sequential_53/dense_102/BiasAddBiasAdd(sequential_53/dense_102/MatMul:product:06sequential_53/dense_102/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b2!
sequential_53/dense_102/BiasAdd�
2sequential_53/batch_normalization_211/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 24
2sequential_53/batch_normalization_211/LogicalAnd/x�
2sequential_53/batch_normalization_211/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z24
2sequential_53/batch_normalization_211/LogicalAnd/y�
0sequential_53/batch_normalization_211/LogicalAnd
LogicalAnd;sequential_53/batch_normalization_211/LogicalAnd/x:output:0;sequential_53/batch_normalization_211/LogicalAnd/y:output:0*
_output_shapes
: 22
0sequential_53/batch_normalization_211/LogicalAnd�
>sequential_53/batch_normalization_211/batchnorm/ReadVariableOpReadVariableOpGsequential_53_batch_normalization_211_batchnorm_readvariableop_resource*
_output_shapes	
:�b*
dtype02@
>sequential_53/batch_normalization_211/batchnorm/ReadVariableOp�
5sequential_53/batch_normalization_211/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:27
5sequential_53/batch_normalization_211/batchnorm/add/y�
3sequential_53/batch_normalization_211/batchnorm/addAddV2Fsequential_53/batch_normalization_211/batchnorm/ReadVariableOp:value:0>sequential_53/batch_normalization_211/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�b25
3sequential_53/batch_normalization_211/batchnorm/add�
5sequential_53/batch_normalization_211/batchnorm/RsqrtRsqrt7sequential_53/batch_normalization_211/batchnorm/add:z:0*
T0*
_output_shapes	
:�b27
5sequential_53/batch_normalization_211/batchnorm/Rsqrt�
Bsequential_53/batch_normalization_211/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_53_batch_normalization_211_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�b*
dtype02D
Bsequential_53/batch_normalization_211/batchnorm/mul/ReadVariableOp�
3sequential_53/batch_normalization_211/batchnorm/mulMul9sequential_53/batch_normalization_211/batchnorm/Rsqrt:y:0Jsequential_53/batch_normalization_211/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b25
3sequential_53/batch_normalization_211/batchnorm/mul�
5sequential_53/batch_normalization_211/batchnorm/mul_1Mul(sequential_53/dense_102/BiasAdd:output:07sequential_53/batch_normalization_211/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������b27
5sequential_53/batch_normalization_211/batchnorm/mul_1�
@sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_53_batch_normalization_211_batchnorm_readvariableop_1_resource*
_output_shapes	
:�b*
dtype02B
@sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_1�
5sequential_53/batch_normalization_211/batchnorm/mul_2MulHsequential_53/batch_normalization_211/batchnorm/ReadVariableOp_1:value:07sequential_53/batch_normalization_211/batchnorm/mul:z:0*
T0*
_output_shapes	
:�b27
5sequential_53/batch_normalization_211/batchnorm/mul_2�
@sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_53_batch_normalization_211_batchnorm_readvariableop_2_resource*
_output_shapes	
:�b*
dtype02B
@sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_2�
3sequential_53/batch_normalization_211/batchnorm/subSubHsequential_53/batch_normalization_211/batchnorm/ReadVariableOp_2:value:09sequential_53/batch_normalization_211/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�b25
3sequential_53/batch_normalization_211/batchnorm/sub�
5sequential_53/batch_normalization_211/batchnorm/add_1AddV29sequential_53/batch_normalization_211/batchnorm/mul_1:z:07sequential_53/batch_normalization_211/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������b27
5sequential_53/batch_normalization_211/batchnorm/add_1�
'sequential_53/leaky_re_lu_208/LeakyRelu	LeakyRelu9sequential_53/batch_normalization_211/batchnorm/add_1:z:0*(
_output_shapes
:����������b2)
'sequential_53/leaky_re_lu_208/LeakyRelu�
sequential_53/reshape_26/ShapeShape5sequential_53/leaky_re_lu_208/LeakyRelu:activations:0*
T0*
_output_shapes
:2 
sequential_53/reshape_26/Shape�
,sequential_53/reshape_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_53/reshape_26/strided_slice/stack�
.sequential_53/reshape_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_53/reshape_26/strided_slice/stack_1�
.sequential_53/reshape_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_53/reshape_26/strided_slice/stack_2�
&sequential_53/reshape_26/strided_sliceStridedSlice'sequential_53/reshape_26/Shape:output:05sequential_53/reshape_26/strided_slice/stack:output:07sequential_53/reshape_26/strided_slice/stack_1:output:07sequential_53/reshape_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_53/reshape_26/strided_slice�
(sequential_53/reshape_26/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_53/reshape_26/Reshape/shape/1�
(sequential_53/reshape_26/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_53/reshape_26/Reshape/shape/2�
(sequential_53/reshape_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :�2*
(sequential_53/reshape_26/Reshape/shape/3�
&sequential_53/reshape_26/Reshape/shapePack/sequential_53/reshape_26/strided_slice:output:01sequential_53/reshape_26/Reshape/shape/1:output:01sequential_53/reshape_26/Reshape/shape/2:output:01sequential_53/reshape_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_53/reshape_26/Reshape/shape�
 sequential_53/reshape_26/ReshapeReshape5sequential_53/leaky_re_lu_208/LeakyRelu:activations:0/sequential_53/reshape_26/Reshape/shape:output:0*
T0*0
_output_shapes
:����������2"
 sequential_53/reshape_26/Reshape�
'sequential_53/conv2d_transpose_78/ShapeShape)sequential_53/reshape_26/Reshape:output:0*
T0*
_output_shapes
:2)
'sequential_53/conv2d_transpose_78/Shape�
5sequential_53/conv2d_transpose_78/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_53/conv2d_transpose_78/strided_slice/stack�
7sequential_53/conv2d_transpose_78/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_78/strided_slice/stack_1�
7sequential_53/conv2d_transpose_78/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_78/strided_slice/stack_2�
/sequential_53/conv2d_transpose_78/strided_sliceStridedSlice0sequential_53/conv2d_transpose_78/Shape:output:0>sequential_53/conv2d_transpose_78/strided_slice/stack:output:0@sequential_53/conv2d_transpose_78/strided_slice/stack_1:output:0@sequential_53/conv2d_transpose_78/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_53/conv2d_transpose_78/strided_slice�
7sequential_53/conv2d_transpose_78/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_78/strided_slice_1/stack�
9sequential_53/conv2d_transpose_78/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_78/strided_slice_1/stack_1�
9sequential_53/conv2d_transpose_78/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_78/strided_slice_1/stack_2�
1sequential_53/conv2d_transpose_78/strided_slice_1StridedSlice0sequential_53/conv2d_transpose_78/Shape:output:0@sequential_53/conv2d_transpose_78/strided_slice_1/stack:output:0Bsequential_53/conv2d_transpose_78/strided_slice_1/stack_1:output:0Bsequential_53/conv2d_transpose_78/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_53/conv2d_transpose_78/strided_slice_1�
7sequential_53/conv2d_transpose_78/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_78/strided_slice_2/stack�
9sequential_53/conv2d_transpose_78/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_78/strided_slice_2/stack_1�
9sequential_53/conv2d_transpose_78/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_78/strided_slice_2/stack_2�
1sequential_53/conv2d_transpose_78/strided_slice_2StridedSlice0sequential_53/conv2d_transpose_78/Shape:output:0@sequential_53/conv2d_transpose_78/strided_slice_2/stack:output:0Bsequential_53/conv2d_transpose_78/strided_slice_2/stack_1:output:0Bsequential_53/conv2d_transpose_78/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_53/conv2d_transpose_78/strided_slice_2�
'sequential_53/conv2d_transpose_78/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_53/conv2d_transpose_78/mul/y�
%sequential_53/conv2d_transpose_78/mulMul:sequential_53/conv2d_transpose_78/strided_slice_1:output:00sequential_53/conv2d_transpose_78/mul/y:output:0*
T0*
_output_shapes
: 2'
%sequential_53/conv2d_transpose_78/mul�
)sequential_53/conv2d_transpose_78/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_53/conv2d_transpose_78/mul_1/y�
'sequential_53/conv2d_transpose_78/mul_1Mul:sequential_53/conv2d_transpose_78/strided_slice_2:output:02sequential_53/conv2d_transpose_78/mul_1/y:output:0*
T0*
_output_shapes
: 2)
'sequential_53/conv2d_transpose_78/mul_1�
)sequential_53/conv2d_transpose_78/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2+
)sequential_53/conv2d_transpose_78/stack/3�
'sequential_53/conv2d_transpose_78/stackPack8sequential_53/conv2d_transpose_78/strided_slice:output:0)sequential_53/conv2d_transpose_78/mul:z:0+sequential_53/conv2d_transpose_78/mul_1:z:02sequential_53/conv2d_transpose_78/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_53/conv2d_transpose_78/stack�
7sequential_53/conv2d_transpose_78/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_53/conv2d_transpose_78/strided_slice_3/stack�
9sequential_53/conv2d_transpose_78/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_78/strided_slice_3/stack_1�
9sequential_53/conv2d_transpose_78/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_78/strided_slice_3/stack_2�
1sequential_53/conv2d_transpose_78/strided_slice_3StridedSlice0sequential_53/conv2d_transpose_78/stack:output:0@sequential_53/conv2d_transpose_78/strided_slice_3/stack:output:0Bsequential_53/conv2d_transpose_78/strided_slice_3/stack_1:output:0Bsequential_53/conv2d_transpose_78/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_53/conv2d_transpose_78/strided_slice_3�
Asequential_53/conv2d_transpose_78/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_53_conv2d_transpose_78_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02C
Asequential_53/conv2d_transpose_78/conv2d_transpose/ReadVariableOp�
2sequential_53/conv2d_transpose_78/conv2d_transposeConv2DBackpropInput0sequential_53/conv2d_transpose_78/stack:output:0Isequential_53/conv2d_transpose_78/conv2d_transpose/ReadVariableOp:value:0)sequential_53/reshape_26/Reshape:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
24
2sequential_53/conv2d_transpose_78/conv2d_transpose�
2sequential_53/batch_normalization_212/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 24
2sequential_53/batch_normalization_212/LogicalAnd/x�
2sequential_53/batch_normalization_212/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z24
2sequential_53/batch_normalization_212/LogicalAnd/y�
0sequential_53/batch_normalization_212/LogicalAnd
LogicalAnd;sequential_53/batch_normalization_212/LogicalAnd/x:output:0;sequential_53/batch_normalization_212/LogicalAnd/y:output:0*
_output_shapes
: 22
0sequential_53/batch_normalization_212/LogicalAnd�
4sequential_53/batch_normalization_212/ReadVariableOpReadVariableOp=sequential_53_batch_normalization_212_readvariableop_resource*
_output_shapes	
:�*
dtype026
4sequential_53/batch_normalization_212/ReadVariableOp�
6sequential_53/batch_normalization_212/ReadVariableOp_1ReadVariableOp?sequential_53_batch_normalization_212_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6sequential_53/batch_normalization_212/ReadVariableOp_1�
Esequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_53_batch_normalization_212_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02G
Esequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOp�
Gsequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_53_batch_normalization_212_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02I
Gsequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOp_1�
6sequential_53/batch_normalization_212/FusedBatchNormV3FusedBatchNormV3;sequential_53/conv2d_transpose_78/conv2d_transpose:output:0<sequential_53/batch_normalization_212/ReadVariableOp:value:0>sequential_53/batch_normalization_212/ReadVariableOp_1:value:0Msequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOp:value:0Osequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 28
6sequential_53/batch_normalization_212/FusedBatchNormV3�
+sequential_53/batch_normalization_212/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2-
+sequential_53/batch_normalization_212/Const�
'sequential_53/leaky_re_lu_209/LeakyRelu	LeakyRelu:sequential_53/batch_normalization_212/FusedBatchNormV3:y:0*0
_output_shapes
:����������2)
'sequential_53/leaky_re_lu_209/LeakyRelu�
'sequential_53/conv2d_transpose_79/ShapeShape5sequential_53/leaky_re_lu_209/LeakyRelu:activations:0*
T0*
_output_shapes
:2)
'sequential_53/conv2d_transpose_79/Shape�
5sequential_53/conv2d_transpose_79/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_53/conv2d_transpose_79/strided_slice/stack�
7sequential_53/conv2d_transpose_79/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_79/strided_slice/stack_1�
7sequential_53/conv2d_transpose_79/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_79/strided_slice/stack_2�
/sequential_53/conv2d_transpose_79/strided_sliceStridedSlice0sequential_53/conv2d_transpose_79/Shape:output:0>sequential_53/conv2d_transpose_79/strided_slice/stack:output:0@sequential_53/conv2d_transpose_79/strided_slice/stack_1:output:0@sequential_53/conv2d_transpose_79/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_53/conv2d_transpose_79/strided_slice�
7sequential_53/conv2d_transpose_79/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_79/strided_slice_1/stack�
9sequential_53/conv2d_transpose_79/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_79/strided_slice_1/stack_1�
9sequential_53/conv2d_transpose_79/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_79/strided_slice_1/stack_2�
1sequential_53/conv2d_transpose_79/strided_slice_1StridedSlice0sequential_53/conv2d_transpose_79/Shape:output:0@sequential_53/conv2d_transpose_79/strided_slice_1/stack:output:0Bsequential_53/conv2d_transpose_79/strided_slice_1/stack_1:output:0Bsequential_53/conv2d_transpose_79/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_53/conv2d_transpose_79/strided_slice_1�
7sequential_53/conv2d_transpose_79/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_79/strided_slice_2/stack�
9sequential_53/conv2d_transpose_79/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_79/strided_slice_2/stack_1�
9sequential_53/conv2d_transpose_79/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_79/strided_slice_2/stack_2�
1sequential_53/conv2d_transpose_79/strided_slice_2StridedSlice0sequential_53/conv2d_transpose_79/Shape:output:0@sequential_53/conv2d_transpose_79/strided_slice_2/stack:output:0Bsequential_53/conv2d_transpose_79/strided_slice_2/stack_1:output:0Bsequential_53/conv2d_transpose_79/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_53/conv2d_transpose_79/strided_slice_2�
'sequential_53/conv2d_transpose_79/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_53/conv2d_transpose_79/mul/y�
%sequential_53/conv2d_transpose_79/mulMul:sequential_53/conv2d_transpose_79/strided_slice_1:output:00sequential_53/conv2d_transpose_79/mul/y:output:0*
T0*
_output_shapes
: 2'
%sequential_53/conv2d_transpose_79/mul�
)sequential_53/conv2d_transpose_79/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_53/conv2d_transpose_79/mul_1/y�
'sequential_53/conv2d_transpose_79/mul_1Mul:sequential_53/conv2d_transpose_79/strided_slice_2:output:02sequential_53/conv2d_transpose_79/mul_1/y:output:0*
T0*
_output_shapes
: 2)
'sequential_53/conv2d_transpose_79/mul_1�
)sequential_53/conv2d_transpose_79/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2+
)sequential_53/conv2d_transpose_79/stack/3�
'sequential_53/conv2d_transpose_79/stackPack8sequential_53/conv2d_transpose_79/strided_slice:output:0)sequential_53/conv2d_transpose_79/mul:z:0+sequential_53/conv2d_transpose_79/mul_1:z:02sequential_53/conv2d_transpose_79/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_53/conv2d_transpose_79/stack�
7sequential_53/conv2d_transpose_79/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_53/conv2d_transpose_79/strided_slice_3/stack�
9sequential_53/conv2d_transpose_79/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_79/strided_slice_3/stack_1�
9sequential_53/conv2d_transpose_79/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_79/strided_slice_3/stack_2�
1sequential_53/conv2d_transpose_79/strided_slice_3StridedSlice0sequential_53/conv2d_transpose_79/stack:output:0@sequential_53/conv2d_transpose_79/strided_slice_3/stack:output:0Bsequential_53/conv2d_transpose_79/strided_slice_3/stack_1:output:0Bsequential_53/conv2d_transpose_79/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_53/conv2d_transpose_79/strided_slice_3�
Asequential_53/conv2d_transpose_79/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_53_conv2d_transpose_79_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02C
Asequential_53/conv2d_transpose_79/conv2d_transpose/ReadVariableOp�
2sequential_53/conv2d_transpose_79/conv2d_transposeConv2DBackpropInput0sequential_53/conv2d_transpose_79/stack:output:0Isequential_53/conv2d_transpose_79/conv2d_transpose/ReadVariableOp:value:05sequential_53/leaky_re_lu_209/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
24
2sequential_53/conv2d_transpose_79/conv2d_transpose�
2sequential_53/batch_normalization_213/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 24
2sequential_53/batch_normalization_213/LogicalAnd/x�
2sequential_53/batch_normalization_213/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z24
2sequential_53/batch_normalization_213/LogicalAnd/y�
0sequential_53/batch_normalization_213/LogicalAnd
LogicalAnd;sequential_53/batch_normalization_213/LogicalAnd/x:output:0;sequential_53/batch_normalization_213/LogicalAnd/y:output:0*
_output_shapes
: 22
0sequential_53/batch_normalization_213/LogicalAnd�
4sequential_53/batch_normalization_213/ReadVariableOpReadVariableOp=sequential_53_batch_normalization_213_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential_53/batch_normalization_213/ReadVariableOp�
6sequential_53/batch_normalization_213/ReadVariableOp_1ReadVariableOp?sequential_53_batch_normalization_213_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6sequential_53/batch_normalization_213/ReadVariableOp_1�
Esequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_53_batch_normalization_213_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02G
Esequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOp�
Gsequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_53_batch_normalization_213_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02I
Gsequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOp_1�
6sequential_53/batch_normalization_213/FusedBatchNormV3FusedBatchNormV3;sequential_53/conv2d_transpose_79/conv2d_transpose:output:0<sequential_53/batch_normalization_213/ReadVariableOp:value:0>sequential_53/batch_normalization_213/ReadVariableOp_1:value:0Msequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOp:value:0Osequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 28
6sequential_53/batch_normalization_213/FusedBatchNormV3�
+sequential_53/batch_normalization_213/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2-
+sequential_53/batch_normalization_213/Const�
'sequential_53/leaky_re_lu_210/LeakyRelu	LeakyRelu:sequential_53/batch_normalization_213/FusedBatchNormV3:y:0*/
_output_shapes
:���������@2)
'sequential_53/leaky_re_lu_210/LeakyRelu�
'sequential_53/conv2d_transpose_80/ShapeShape5sequential_53/leaky_re_lu_210/LeakyRelu:activations:0*
T0*
_output_shapes
:2)
'sequential_53/conv2d_transpose_80/Shape�
5sequential_53/conv2d_transpose_80/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_53/conv2d_transpose_80/strided_slice/stack�
7sequential_53/conv2d_transpose_80/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_80/strided_slice/stack_1�
7sequential_53/conv2d_transpose_80/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_80/strided_slice/stack_2�
/sequential_53/conv2d_transpose_80/strided_sliceStridedSlice0sequential_53/conv2d_transpose_80/Shape:output:0>sequential_53/conv2d_transpose_80/strided_slice/stack:output:0@sequential_53/conv2d_transpose_80/strided_slice/stack_1:output:0@sequential_53/conv2d_transpose_80/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_53/conv2d_transpose_80/strided_slice�
7sequential_53/conv2d_transpose_80/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_80/strided_slice_1/stack�
9sequential_53/conv2d_transpose_80/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_80/strided_slice_1/stack_1�
9sequential_53/conv2d_transpose_80/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_80/strided_slice_1/stack_2�
1sequential_53/conv2d_transpose_80/strided_slice_1StridedSlice0sequential_53/conv2d_transpose_80/Shape:output:0@sequential_53/conv2d_transpose_80/strided_slice_1/stack:output:0Bsequential_53/conv2d_transpose_80/strided_slice_1/stack_1:output:0Bsequential_53/conv2d_transpose_80/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_53/conv2d_transpose_80/strided_slice_1�
7sequential_53/conv2d_transpose_80/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_53/conv2d_transpose_80/strided_slice_2/stack�
9sequential_53/conv2d_transpose_80/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_80/strided_slice_2/stack_1�
9sequential_53/conv2d_transpose_80/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_80/strided_slice_2/stack_2�
1sequential_53/conv2d_transpose_80/strided_slice_2StridedSlice0sequential_53/conv2d_transpose_80/Shape:output:0@sequential_53/conv2d_transpose_80/strided_slice_2/stack:output:0Bsequential_53/conv2d_transpose_80/strided_slice_2/stack_1:output:0Bsequential_53/conv2d_transpose_80/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_53/conv2d_transpose_80/strided_slice_2�
'sequential_53/conv2d_transpose_80/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_53/conv2d_transpose_80/mul/y�
%sequential_53/conv2d_transpose_80/mulMul:sequential_53/conv2d_transpose_80/strided_slice_1:output:00sequential_53/conv2d_transpose_80/mul/y:output:0*
T0*
_output_shapes
: 2'
%sequential_53/conv2d_transpose_80/mul�
)sequential_53/conv2d_transpose_80/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_53/conv2d_transpose_80/mul_1/y�
'sequential_53/conv2d_transpose_80/mul_1Mul:sequential_53/conv2d_transpose_80/strided_slice_2:output:02sequential_53/conv2d_transpose_80/mul_1/y:output:0*
T0*
_output_shapes
: 2)
'sequential_53/conv2d_transpose_80/mul_1�
)sequential_53/conv2d_transpose_80/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_53/conv2d_transpose_80/stack/3�
'sequential_53/conv2d_transpose_80/stackPack8sequential_53/conv2d_transpose_80/strided_slice:output:0)sequential_53/conv2d_transpose_80/mul:z:0+sequential_53/conv2d_transpose_80/mul_1:z:02sequential_53/conv2d_transpose_80/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_53/conv2d_transpose_80/stack�
7sequential_53/conv2d_transpose_80/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_53/conv2d_transpose_80/strided_slice_3/stack�
9sequential_53/conv2d_transpose_80/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_80/strided_slice_3/stack_1�
9sequential_53/conv2d_transpose_80/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_53/conv2d_transpose_80/strided_slice_3/stack_2�
1sequential_53/conv2d_transpose_80/strided_slice_3StridedSlice0sequential_53/conv2d_transpose_80/stack:output:0@sequential_53/conv2d_transpose_80/strided_slice_3/stack:output:0Bsequential_53/conv2d_transpose_80/strided_slice_3/stack_1:output:0Bsequential_53/conv2d_transpose_80/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_53/conv2d_transpose_80/strided_slice_3�
Asequential_53/conv2d_transpose_80/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_53_conv2d_transpose_80_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02C
Asequential_53/conv2d_transpose_80/conv2d_transpose/ReadVariableOp�
2sequential_53/conv2d_transpose_80/conv2d_transposeConv2DBackpropInput0sequential_53/conv2d_transpose_80/stack:output:0Isequential_53/conv2d_transpose_80/conv2d_transpose/ReadVariableOp:value:05sequential_53/leaky_re_lu_210/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
24
2sequential_53/conv2d_transpose_80/conv2d_transpose�
&sequential_53/conv2d_transpose_80/TanhTanh;sequential_53/conv2d_transpose_80/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������2(
&sequential_53/conv2d_transpose_80/Tanh�
IdentityIdentity*sequential_53/conv2d_transpose_80/Tanh:y:0?^sequential_53/batch_normalization_210/batchnorm/ReadVariableOpA^sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_1A^sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_2C^sequential_53/batch_normalization_210/batchnorm/mul/ReadVariableOp?^sequential_53/batch_normalization_211/batchnorm/ReadVariableOpA^sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_1A^sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_2C^sequential_53/batch_normalization_211/batchnorm/mul/ReadVariableOpF^sequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOpH^sequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOp_15^sequential_53/batch_normalization_212/ReadVariableOp7^sequential_53/batch_normalization_212/ReadVariableOp_1F^sequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOpH^sequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOp_15^sequential_53/batch_normalization_213/ReadVariableOp7^sequential_53/batch_normalization_213/ReadVariableOp_1B^sequential_53/conv2d_transpose_78/conv2d_transpose/ReadVariableOpB^sequential_53/conv2d_transpose_79/conv2d_transpose/ReadVariableOpB^sequential_53/conv2d_transpose_80/conv2d_transpose/ReadVariableOp/^sequential_53/dense_101/BiasAdd/ReadVariableOp.^sequential_53/dense_101/MatMul/ReadVariableOp/^sequential_53/dense_102/BiasAdd/ReadVariableOp.^sequential_53/dense_102/MatMul/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::2�
>sequential_53/batch_normalization_210/batchnorm/ReadVariableOp>sequential_53/batch_normalization_210/batchnorm/ReadVariableOp2�
@sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_1@sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_12�
@sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_2@sequential_53/batch_normalization_210/batchnorm/ReadVariableOp_22�
Bsequential_53/batch_normalization_210/batchnorm/mul/ReadVariableOpBsequential_53/batch_normalization_210/batchnorm/mul/ReadVariableOp2�
>sequential_53/batch_normalization_211/batchnorm/ReadVariableOp>sequential_53/batch_normalization_211/batchnorm/ReadVariableOp2�
@sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_1@sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_12�
@sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_2@sequential_53/batch_normalization_211/batchnorm/ReadVariableOp_22�
Bsequential_53/batch_normalization_211/batchnorm/mul/ReadVariableOpBsequential_53/batch_normalization_211/batchnorm/mul/ReadVariableOp2�
Esequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOpEsequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOp2�
Gsequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOp_1Gsequential_53/batch_normalization_212/FusedBatchNormV3/ReadVariableOp_12l
4sequential_53/batch_normalization_212/ReadVariableOp4sequential_53/batch_normalization_212/ReadVariableOp2p
6sequential_53/batch_normalization_212/ReadVariableOp_16sequential_53/batch_normalization_212/ReadVariableOp_12�
Esequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOpEsequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOp2�
Gsequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOp_1Gsequential_53/batch_normalization_213/FusedBatchNormV3/ReadVariableOp_12l
4sequential_53/batch_normalization_213/ReadVariableOp4sequential_53/batch_normalization_213/ReadVariableOp2p
6sequential_53/batch_normalization_213/ReadVariableOp_16sequential_53/batch_normalization_213/ReadVariableOp_12�
Asequential_53/conv2d_transpose_78/conv2d_transpose/ReadVariableOpAsequential_53/conv2d_transpose_78/conv2d_transpose/ReadVariableOp2�
Asequential_53/conv2d_transpose_79/conv2d_transpose/ReadVariableOpAsequential_53/conv2d_transpose_79/conv2d_transpose/ReadVariableOp2�
Asequential_53/conv2d_transpose_80/conv2d_transpose/ReadVariableOpAsequential_53/conv2d_transpose_80/conv2d_transpose/ReadVariableOp2`
.sequential_53/dense_101/BiasAdd/ReadVariableOp.sequential_53/dense_101/BiasAdd/ReadVariableOp2^
-sequential_53/dense_101/MatMul/ReadVariableOp-sequential_53/dense_101/MatMul/ReadVariableOp2`
.sequential_53/dense_102/BiasAdd/ReadVariableOp.sequential_53/dense_102/BiasAdd/ReadVariableOp2^
-sequential_53/dense_102/MatMul/ReadVariableOp-sequential_53/dense_102/MatMul/ReadVariableOp:/ +
)
_user_specified_namedense_101_input
�/
�
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_838277

inputs
assignmovingavg_838252
assignmovingavg_1_838258)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/838252*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_838252*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/838252*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/838252*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_838252AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/838252*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/838258*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_838258*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/838258*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/838258*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_838258AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/838258*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�R
�
I__inference_sequential_53_layer_call_and_return_conditional_losses_839145

inputs,
(dense_101_statefulpartitionedcall_args_1,
(dense_101_statefulpartitionedcall_args_2:
6batch_normalization_210_statefulpartitionedcall_args_1:
6batch_normalization_210_statefulpartitionedcall_args_2:
6batch_normalization_210_statefulpartitionedcall_args_3:
6batch_normalization_210_statefulpartitionedcall_args_4,
(dense_102_statefulpartitionedcall_args_1,
(dense_102_statefulpartitionedcall_args_2:
6batch_normalization_211_statefulpartitionedcall_args_1:
6batch_normalization_211_statefulpartitionedcall_args_2:
6batch_normalization_211_statefulpartitionedcall_args_3:
6batch_normalization_211_statefulpartitionedcall_args_46
2conv2d_transpose_78_statefulpartitionedcall_args_1:
6batch_normalization_212_statefulpartitionedcall_args_1:
6batch_normalization_212_statefulpartitionedcall_args_2:
6batch_normalization_212_statefulpartitionedcall_args_3:
6batch_normalization_212_statefulpartitionedcall_args_46
2conv2d_transpose_79_statefulpartitionedcall_args_1:
6batch_normalization_213_statefulpartitionedcall_args_1:
6batch_normalization_213_statefulpartitionedcall_args_2:
6batch_normalization_213_statefulpartitionedcall_args_3:
6batch_normalization_213_statefulpartitionedcall_args_46
2conv2d_transpose_80_statefulpartitionedcall_args_1
identity��/batch_normalization_210/StatefulPartitionedCall�/batch_normalization_211/StatefulPartitionedCall�/batch_normalization_212/StatefulPartitionedCall�/batch_normalization_213/StatefulPartitionedCall�+conv2d_transpose_78/StatefulPartitionedCall�+conv2d_transpose_79/StatefulPartitionedCall�+conv2d_transpose_80/StatefulPartitionedCall�!dense_101/StatefulPartitionedCall�!dense_102/StatefulPartitionedCall�
!dense_101/StatefulPartitionedCallStatefulPartitionedCallinputs(dense_101_statefulpartitionedcall_args_1(dense_101_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_101_layer_call_and_return_conditional_losses_8388532#
!dense_101/StatefulPartitionedCall�
/batch_normalization_210/StatefulPartitionedCallStatefulPartitionedCall*dense_101/StatefulPartitionedCall:output:06batch_normalization_210_statefulpartitionedcall_args_16batch_normalization_210_statefulpartitionedcall_args_26batch_normalization_210_statefulpartitionedcall_args_36batch_normalization_210_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_83827721
/batch_normalization_210/StatefulPartitionedCall�
leaky_re_lu_207/PartitionedCallPartitionedCall8batch_normalization_210/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_207_layer_call_and_return_conditional_losses_8388932!
leaky_re_lu_207/PartitionedCall�
!dense_102/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_207/PartitionedCall:output:0(dense_102_statefulpartitionedcall_args_1(dense_102_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_102_layer_call_and_return_conditional_losses_8389112#
!dense_102/StatefulPartitionedCall�
/batch_normalization_211/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:06batch_normalization_211_statefulpartitionedcall_args_16batch_normalization_211_statefulpartitionedcall_args_26batch_normalization_211_statefulpartitionedcall_args_36batch_normalization_211_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_83842121
/batch_normalization_211/StatefulPartitionedCall�
leaky_re_lu_208/PartitionedCallPartitionedCall8batch_normalization_211/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_208_layer_call_and_return_conditional_losses_8389512!
leaky_re_lu_208/PartitionedCall�
reshape_26/PartitionedCallPartitionedCall(leaky_re_lu_208/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_reshape_26_layer_call_and_return_conditional_losses_8389732
reshape_26/PartitionedCall�
+conv2d_transpose_78/StatefulPartitionedCallStatefulPartitionedCall#reshape_26/PartitionedCall:output:02conv2d_transpose_78_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_78_layer_call_and_return_conditional_losses_8384912-
+conv2d_transpose_78/StatefulPartitionedCall�
/batch_normalization_212/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_78/StatefulPartitionedCall:output:06batch_normalization_212_statefulpartitionedcall_args_16batch_normalization_212_statefulpartitionedcall_args_26batch_normalization_212_statefulpartitionedcall_args_36batch_normalization_212_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_83859221
/batch_normalization_212/StatefulPartitionedCall�
leaky_re_lu_209/PartitionedCallPartitionedCall8batch_normalization_212/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_209_layer_call_and_return_conditional_losses_8390112!
leaky_re_lu_209/PartitionedCall�
+conv2d_transpose_79/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_209/PartitionedCall:output:02conv2d_transpose_79_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_79_layer_call_and_return_conditional_losses_8386612-
+conv2d_transpose_79/StatefulPartitionedCall�
/batch_normalization_213/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_79/StatefulPartitionedCall:output:06batch_normalization_213_statefulpartitionedcall_args_16batch_normalization_213_statefulpartitionedcall_args_26batch_normalization_213_statefulpartitionedcall_args_36batch_normalization_213_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_83876221
/batch_normalization_213/StatefulPartitionedCall�
leaky_re_lu_210/PartitionedCallPartitionedCall8batch_normalization_213/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_210_layer_call_and_return_conditional_losses_8390492!
leaky_re_lu_210/PartitionedCall�
+conv2d_transpose_80/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_210/PartitionedCall:output:02conv2d_transpose_80_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_8388322-
+conv2d_transpose_80/StatefulPartitionedCall�
IdentityIdentity4conv2d_transpose_80/StatefulPartitionedCall:output:00^batch_normalization_210/StatefulPartitionedCall0^batch_normalization_211/StatefulPartitionedCall0^batch_normalization_212/StatefulPartitionedCall0^batch_normalization_213/StatefulPartitionedCall,^conv2d_transpose_78/StatefulPartitionedCall,^conv2d_transpose_79/StatefulPartitionedCall,^conv2d_transpose_80/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::2b
/batch_normalization_210/StatefulPartitionedCall/batch_normalization_210/StatefulPartitionedCall2b
/batch_normalization_211/StatefulPartitionedCall/batch_normalization_211/StatefulPartitionedCall2b
/batch_normalization_212/StatefulPartitionedCall/batch_normalization_212/StatefulPartitionedCall2b
/batch_normalization_213/StatefulPartitionedCall/batch_normalization_213/StatefulPartitionedCall2Z
+conv2d_transpose_78/StatefulPartitionedCall+conv2d_transpose_78/StatefulPartitionedCall2Z
+conv2d_transpose_79/StatefulPartitionedCall+conv2d_transpose_79/StatefulPartitionedCall2Z
+conv2d_transpose_80/StatefulPartitionedCall+conv2d_transpose_80/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�/
�
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_839935

inputs
assignmovingavg_839910
assignmovingavg_1_839916)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/839910*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_839910*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/839910*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/839910*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_839910AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/839910*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/839916*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_839916*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/839916*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/839916*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_839916AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/839916*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�R
�
I__inference_sequential_53_layer_call_and_return_conditional_losses_839101
dense_101_input,
(dense_101_statefulpartitionedcall_args_1,
(dense_101_statefulpartitionedcall_args_2:
6batch_normalization_210_statefulpartitionedcall_args_1:
6batch_normalization_210_statefulpartitionedcall_args_2:
6batch_normalization_210_statefulpartitionedcall_args_3:
6batch_normalization_210_statefulpartitionedcall_args_4,
(dense_102_statefulpartitionedcall_args_1,
(dense_102_statefulpartitionedcall_args_2:
6batch_normalization_211_statefulpartitionedcall_args_1:
6batch_normalization_211_statefulpartitionedcall_args_2:
6batch_normalization_211_statefulpartitionedcall_args_3:
6batch_normalization_211_statefulpartitionedcall_args_46
2conv2d_transpose_78_statefulpartitionedcall_args_1:
6batch_normalization_212_statefulpartitionedcall_args_1:
6batch_normalization_212_statefulpartitionedcall_args_2:
6batch_normalization_212_statefulpartitionedcall_args_3:
6batch_normalization_212_statefulpartitionedcall_args_46
2conv2d_transpose_79_statefulpartitionedcall_args_1:
6batch_normalization_213_statefulpartitionedcall_args_1:
6batch_normalization_213_statefulpartitionedcall_args_2:
6batch_normalization_213_statefulpartitionedcall_args_3:
6batch_normalization_213_statefulpartitionedcall_args_46
2conv2d_transpose_80_statefulpartitionedcall_args_1
identity��/batch_normalization_210/StatefulPartitionedCall�/batch_normalization_211/StatefulPartitionedCall�/batch_normalization_212/StatefulPartitionedCall�/batch_normalization_213/StatefulPartitionedCall�+conv2d_transpose_78/StatefulPartitionedCall�+conv2d_transpose_79/StatefulPartitionedCall�+conv2d_transpose_80/StatefulPartitionedCall�!dense_101/StatefulPartitionedCall�!dense_102/StatefulPartitionedCall�
!dense_101/StatefulPartitionedCallStatefulPartitionedCalldense_101_input(dense_101_statefulpartitionedcall_args_1(dense_101_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_101_layer_call_and_return_conditional_losses_8388532#
!dense_101/StatefulPartitionedCall�
/batch_normalization_210/StatefulPartitionedCallStatefulPartitionedCall*dense_101/StatefulPartitionedCall:output:06batch_normalization_210_statefulpartitionedcall_args_16batch_normalization_210_statefulpartitionedcall_args_26batch_normalization_210_statefulpartitionedcall_args_36batch_normalization_210_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_83830921
/batch_normalization_210/StatefulPartitionedCall�
leaky_re_lu_207/PartitionedCallPartitionedCall8batch_normalization_210/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_207_layer_call_and_return_conditional_losses_8388932!
leaky_re_lu_207/PartitionedCall�
!dense_102/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_207/PartitionedCall:output:0(dense_102_statefulpartitionedcall_args_1(dense_102_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_102_layer_call_and_return_conditional_losses_8389112#
!dense_102/StatefulPartitionedCall�
/batch_normalization_211/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:06batch_normalization_211_statefulpartitionedcall_args_16batch_normalization_211_statefulpartitionedcall_args_26batch_normalization_211_statefulpartitionedcall_args_36batch_normalization_211_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_83845321
/batch_normalization_211/StatefulPartitionedCall�
leaky_re_lu_208/PartitionedCallPartitionedCall8batch_normalization_211/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_208_layer_call_and_return_conditional_losses_8389512!
leaky_re_lu_208/PartitionedCall�
reshape_26/PartitionedCallPartitionedCall(leaky_re_lu_208/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_reshape_26_layer_call_and_return_conditional_losses_8389732
reshape_26/PartitionedCall�
+conv2d_transpose_78/StatefulPartitionedCallStatefulPartitionedCall#reshape_26/PartitionedCall:output:02conv2d_transpose_78_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_78_layer_call_and_return_conditional_losses_8384912-
+conv2d_transpose_78/StatefulPartitionedCall�
/batch_normalization_212/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_78/StatefulPartitionedCall:output:06batch_normalization_212_statefulpartitionedcall_args_16batch_normalization_212_statefulpartitionedcall_args_26batch_normalization_212_statefulpartitionedcall_args_36batch_normalization_212_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_83862321
/batch_normalization_212/StatefulPartitionedCall�
leaky_re_lu_209/PartitionedCallPartitionedCall8batch_normalization_212/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_209_layer_call_and_return_conditional_losses_8390112!
leaky_re_lu_209/PartitionedCall�
+conv2d_transpose_79/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_209/PartitionedCall:output:02conv2d_transpose_79_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_79_layer_call_and_return_conditional_losses_8386612-
+conv2d_transpose_79/StatefulPartitionedCall�
/batch_normalization_213/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_79/StatefulPartitionedCall:output:06batch_normalization_213_statefulpartitionedcall_args_16batch_normalization_213_statefulpartitionedcall_args_26batch_normalization_213_statefulpartitionedcall_args_36batch_normalization_213_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_83879321
/batch_normalization_213/StatefulPartitionedCall�
leaky_re_lu_210/PartitionedCallPartitionedCall8batch_normalization_213/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_210_layer_call_and_return_conditional_losses_8390492!
leaky_re_lu_210/PartitionedCall�
+conv2d_transpose_80/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_210/PartitionedCall:output:02conv2d_transpose_80_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_8388322-
+conv2d_transpose_80/StatefulPartitionedCall�
IdentityIdentity4conv2d_transpose_80/StatefulPartitionedCall:output:00^batch_normalization_210/StatefulPartitionedCall0^batch_normalization_211/StatefulPartitionedCall0^batch_normalization_212/StatefulPartitionedCall0^batch_normalization_213/StatefulPartitionedCall,^conv2d_transpose_78/StatefulPartitionedCall,^conv2d_transpose_79/StatefulPartitionedCall,^conv2d_transpose_80/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::2b
/batch_normalization_210/StatefulPartitionedCall/batch_normalization_210/StatefulPartitionedCall2b
/batch_normalization_211/StatefulPartitionedCall/batch_normalization_211/StatefulPartitionedCall2b
/batch_normalization_212/StatefulPartitionedCall/batch_normalization_212/StatefulPartitionedCall2b
/batch_normalization_213/StatefulPartitionedCall/batch_normalization_213/StatefulPartitionedCall2Z
+conv2d_transpose_78/StatefulPartitionedCall+conv2d_transpose_78/StatefulPartitionedCall2Z
+conv2d_transpose_79/StatefulPartitionedCall+conv2d_transpose_79/StatefulPartitionedCall2Z
+conv2d_transpose_80/StatefulPartitionedCall+conv2d_transpose_80/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall:/ +
)
_user_specified_namedense_101_input
�
�
*__inference_dense_102_layer_call_fn_840003

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_102_layer_call_and_return_conditional_losses_8389112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�e
�
"__inference__traced_restore_840514
file_prefix%
!assignvariableop_dense_101_kernel%
!assignvariableop_1_dense_101_bias4
0assignvariableop_2_batch_normalization_210_gamma3
/assignvariableop_3_batch_normalization_210_beta:
6assignvariableop_4_batch_normalization_210_moving_mean>
:assignvariableop_5_batch_normalization_210_moving_variance'
#assignvariableop_6_dense_102_kernel%
!assignvariableop_7_dense_102_bias4
0assignvariableop_8_batch_normalization_211_gamma3
/assignvariableop_9_batch_normalization_211_beta;
7assignvariableop_10_batch_normalization_211_moving_mean?
;assignvariableop_11_batch_normalization_211_moving_variance2
.assignvariableop_12_conv2d_transpose_78_kernel5
1assignvariableop_13_batch_normalization_212_gamma4
0assignvariableop_14_batch_normalization_212_beta;
7assignvariableop_15_batch_normalization_212_moving_mean?
;assignvariableop_16_batch_normalization_212_moving_variance2
.assignvariableop_17_conv2d_transpose_79_kernel5
1assignvariableop_18_batch_normalization_213_gamma4
0assignvariableop_19_batch_normalization_213_beta;
7assignvariableop_20_batch_normalization_213_moving_mean?
;assignvariableop_21_batch_normalization_213_moving_variance2
.assignvariableop_22_conv2d_transpose_80_kernel
identity_24��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_dense_101_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_101_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_batch_normalization_210_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_batch_normalization_210_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_210_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp:assignvariableop_5_batch_normalization_210_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_102_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_102_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_batch_normalization_211_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_211_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batch_normalization_211_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp;assignvariableop_11_batch_normalization_211_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp.assignvariableop_12_conv2d_transpose_78_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp1assignvariableop_13_batch_normalization_212_gammaIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_212_betaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp7assignvariableop_15_batch_normalization_212_moving_meanIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp;assignvariableop_16_batch_normalization_212_moving_varianceIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_conv2d_transpose_79_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_batch_normalization_213_gammaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp0assignvariableop_19_batch_normalization_213_betaIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp7assignvariableop_20_batch_normalization_213_moving_meanIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp;assignvariableop_21_batch_normalization_213_moving_varianceIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp.assignvariableop_22_conv2d_transpose_80_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23�
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�
�
8__inference_batch_normalization_213_layer_call_fn_840330

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_8387932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_838623

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
4__inference_conv2d_transpose_80_layer_call_fn_838839

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_8388322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+���������������������������@:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_211_layer_call_fn_840119

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_8384532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������b::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_838453

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�b*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�b2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�b2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�b*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������b2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�b*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�b2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�b*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�b2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������b2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������b::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_210_layer_call_fn_839967

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_8382772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
L
0__inference_leaky_re_lu_207_layer_call_fn_839986

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_207_layer_call_and_return_conditional_losses_8388932
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
E__inference_dense_102_layer_call_and_return_conditional_losses_839996

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��b*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�b*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�/
�
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_838421

inputs
assignmovingavg_838396
assignmovingavg_1_838402)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�b*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�b2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������b2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�b*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�b*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�b*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/838396*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_838396*
_output_shapes	
:�b*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/838396*
_output_shapes	
:�b2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/838396*
_output_shapes	
:�b2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_838396AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/838396*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/838402*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_838402*
_output_shapes	
:�b*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/838402*
_output_shapes	
:�b2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/838402*
_output_shapes	
:�b2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_838402AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/838402*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�b2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�b2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�b*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������b2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�b2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�b*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�b2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������b2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������b::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_212_layer_call_fn_840234

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_8386232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
� 
�
O__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_838832

inputs,
(conv2d_transpose_readvariableop_resource
identity��conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
conv2d_transpose{
TanhTanhconv2d_transpose:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Tanh�
IdentityIdentityTanh:y:0 ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+���������������������������@:2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_838309

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_840216

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_210_layer_call_and_return_conditional_losses_840335

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������@2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:& "
 
_user_specified_nameinputs
�
�
.__inference_sequential_53_layer_call_fn_839240
dense_101_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_101_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_53_layer_call_and_return_conditional_losses_8392142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_namedense_101_input
�
�
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_840312

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
G
+__inference_reshape_26_layer_call_fn_840148

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_reshape_26_layer_call_and_return_conditional_losses_8389732
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������b:& "
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_840290

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_840275
assignmovingavg_1_840282
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/840275*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/840275*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_840275*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/840275*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/840275*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_840275AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/840275*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/840282*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/840282*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_840282*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/840282*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/840282*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_840282AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/840282*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
O__inference_conv2d_transpose_79_layer_call_and_return_conditional_losses_838661

inputs,
(conv2d_transpose_readvariableop_resource
identity��conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
conv2d_transpose�
IdentityIdentityconv2d_transpose:output:0 ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,����������������������������:2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
E__inference_dense_101_layer_call_and_return_conditional_losses_839853

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
E__inference_dense_102_layer_call_and_return_conditional_losses_838911

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��b*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�b*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_209_layer_call_and_return_conditional_losses_839011

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:& "
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_210_layer_call_and_return_conditional_losses_839049

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������@2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:& "
 
_user_specified_nameinputs
�/
�
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_840078

inputs
assignmovingavg_840053
assignmovingavg_1_840059)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�b*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�b2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������b2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�b*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�b*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�b*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/840053*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_840053*
_output_shapes	
:�b*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/840053*
_output_shapes	
:�b2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/840053*
_output_shapes	
:�b2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_840053AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/840053*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/840059*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_840059*
_output_shapes	
:�b*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/840059*
_output_shapes	
:�b2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/840059*
_output_shapes	
:�b2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_840059AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/840059*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�b2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�b2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�b*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������b2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�b2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�b*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�b2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������b2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������b::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_839958

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_209_layer_call_and_return_conditional_losses_840239

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:& "
 
_user_specified_nameinputs
�8
�
__inference__traced_save_840433
file_prefix/
+savev2_dense_101_kernel_read_readvariableop-
)savev2_dense_101_bias_read_readvariableop<
8savev2_batch_normalization_210_gamma_read_readvariableop;
7savev2_batch_normalization_210_beta_read_readvariableopB
>savev2_batch_normalization_210_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_210_moving_variance_read_readvariableop/
+savev2_dense_102_kernel_read_readvariableop-
)savev2_dense_102_bias_read_readvariableop<
8savev2_batch_normalization_211_gamma_read_readvariableop;
7savev2_batch_normalization_211_beta_read_readvariableopB
>savev2_batch_normalization_211_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_211_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_78_kernel_read_readvariableop<
8savev2_batch_normalization_212_gamma_read_readvariableop;
7savev2_batch_normalization_212_beta_read_readvariableopB
>savev2_batch_normalization_212_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_212_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_79_kernel_read_readvariableop<
8savev2_batch_normalization_213_gamma_read_readvariableop;
7savev2_batch_normalization_213_beta_read_readvariableopB
>savev2_batch_normalization_213_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_213_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_80_kernel_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_3671ce9d26b14d6d93d611978a952e6d/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_101_kernel_read_readvariableop)savev2_dense_101_bias_read_readvariableop8savev2_batch_normalization_210_gamma_read_readvariableop7savev2_batch_normalization_210_beta_read_readvariableop>savev2_batch_normalization_210_moving_mean_read_readvariableopBsavev2_batch_normalization_210_moving_variance_read_readvariableop+savev2_dense_102_kernel_read_readvariableop)savev2_dense_102_bias_read_readvariableop8savev2_batch_normalization_211_gamma_read_readvariableop7savev2_batch_normalization_211_beta_read_readvariableop>savev2_batch_normalization_211_moving_mean_read_readvariableopBsavev2_batch_normalization_211_moving_variance_read_readvariableop5savev2_conv2d_transpose_78_kernel_read_readvariableop8savev2_batch_normalization_212_gamma_read_readvariableop7savev2_batch_normalization_212_beta_read_readvariableop>savev2_batch_normalization_212_moving_mean_read_readvariableopBsavev2_batch_normalization_212_moving_variance_read_readvariableop5savev2_conv2d_transpose_79_kernel_read_readvariableop8savev2_batch_normalization_213_gamma_read_readvariableop7savev2_batch_normalization_213_beta_read_readvariableop>savev2_batch_normalization_213_moving_mean_read_readvariableopBsavev2_batch_normalization_213_moving_variance_read_readvariableop5savev2_conv2d_transpose_80_kernel_read_readvariableop"/device:CPU:0*
_output_shapes
 *%
dtypes
22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	d�:�:�:�:�:�:
��b:�b:�b:�b:�b:�b:��:�:�:�:�:@�:@:@:@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
�
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_840101

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�b*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�b2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�b2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�b*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������b2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�b*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�b2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�b*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�b2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������b2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������b::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_208_layer_call_and_return_conditional_losses_840124

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:����������b2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������b:& "
 
_user_specified_nameinputs
�
�
*__inference_dense_101_layer_call_fn_839860

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_101_layer_call_and_return_conditional_losses_8388532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_838762

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_838747
assignmovingavg_1_838754
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/838747*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/838747*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_838747*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/838747*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/838747*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_838747AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/838747*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/838754*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/838754*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_838754*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/838754*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/838754*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_838754AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/838754*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
.__inference_sequential_53_layer_call_fn_839171
dense_101_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_101_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_53_layer_call_and_return_conditional_losses_8391452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapesq
o:���������d:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_namedense_101_input
�
�
8__inference_batch_normalization_211_layer_call_fn_840110

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������b*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_8384212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������b2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������b::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
L
0__inference_leaky_re_lu_209_layer_call_fn_840244

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_leaky_re_lu_209_layer_call_and_return_conditional_losses_8390112
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:& "
 
_user_specified_nameinputs
�
�
4__inference_conv2d_transpose_78_layer_call_fn_838498

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,����������������������������*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_78_layer_call_and_return_conditional_losses_8384912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,����������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
dense_101_input8
!serving_default_dense_101_input:0���������dO
conv2d_transpose_808
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�`
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses"�[
_tf_keras_sequential�[{"class_name": "Sequential", "name": "sequential_53", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_53", "layers": [{"class_name": "Dense", "config": {"name": "dense_101", "trainable": true, "batch_input_shape": [null, 100], "dtype": "float32", "units": 1024, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_210", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_207", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dense", "config": {"name": "dense_102", "trainable": true, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_211", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_208", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Reshape", "config": {"name": "reshape_26", "trainable": true, "dtype": "float32", "target_shape": [7, 7, 256]}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_78", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_212", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_209", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_79", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_213", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_210", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_80", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_53", "layers": [{"class_name": "Dense", "config": {"name": "dense_101", "trainable": true, "batch_input_shape": [null, 100], "dtype": "float32", "units": 1024, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_210", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_207", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dense", "config": {"name": "dense_102", "trainable": true, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_211", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_208", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Reshape", "config": {"name": "reshape_26", "trainable": true, "dtype": "float32", "target_shape": [7, 7, 256]}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_78", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_212", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_209", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_79", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_213", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_210", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_80", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "dense_101_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 100], "config": {"batch_input_shape": [null, 100], "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_101_input"}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_101", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 100], "config": {"name": "dense_101", "trainable": true, "batch_input_shape": [null, 100], "dtype": "float32", "units": 1024, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
�
axis
	gamma
beta
moving_mean
moving_variance
 	variables
!regularization_losses
"trainable_variables
#	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_210", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_210", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 1024}}}}
�
$	variables
%regularization_losses
&trainable_variables
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_207", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_207", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_102", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_102", "trainable": true, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}}
�
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4regularization_losses
5trainable_variables
6	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_211", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_211", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 12544}}}}
�
7	variables
8regularization_losses
9trainable_variables
:	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_208", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_208", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�
;	variables
<regularization_losses
=trainable_variables
>	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "reshape_26", "trainable": true, "dtype": "float32", "target_shape": [7, 7, 256]}}
�

?kernel
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_78", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_78", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
�
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_212", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_212", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
�
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_209", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_209", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�

Qkernel
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_79", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_79", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
�
Vaxis
	Wgamma
Xbeta
Ymoving_mean
Zmoving_variance
[	variables
\regularization_losses
]trainable_variables
^	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_213", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_213", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
_	variables
`regularization_losses
atrainable_variables
b	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_210", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_210", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
�

ckernel
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_80", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
0
1
2
3
4
5
(6
)7
/8
09
110
211
?12
E13
F14
G15
H16
Q17
W18
X19
Y20
Z21
c22"
trackable_list_wrapper
�
0
1
2
3
(4
)5
/6
07
?8
E9
F10
Q11
W12
X13
c14"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
hmetrics
trainable_variables

ilayers
regularization_losses
jnon_trainable_variables
klayer_regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
#:!	d�2dense_101/kernel
:�2dense_101/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
lmetrics
regularization_losses

mlayers
trainable_variables
nnon_trainable_variables
olayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*�2batch_normalization_210/gamma
+:)�2batch_normalization_210/beta
4:2� (2#batch_normalization_210/moving_mean
8:6� (2'batch_normalization_210/moving_variance
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
 	variables
pmetrics
!regularization_losses

qlayers
"trainable_variables
rnon_trainable_variables
slayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
$	variables
tmetrics
%regularization_losses

ulayers
&trainable_variables
vnon_trainable_variables
wlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
$:"
��b2dense_102/kernel
:�b2dense_102/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
�
*	variables
xmetrics
+regularization_losses

ylayers
,trainable_variables
znon_trainable_variables
{layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*�b2batch_normalization_211/gamma
+:)�b2batch_normalization_211/beta
4:2�b (2#batch_normalization_211/moving_mean
8:6�b (2'batch_normalization_211/moving_variance
<
/0
01
12
23"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
�
3	variables
|metrics
4regularization_losses

}layers
5trainable_variables
~non_trainable_variables
layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
7	variables
�metrics
8regularization_losses
�layers
9trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
;	variables
�metrics
<regularization_losses
�layers
=trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
6:4��2conv2d_transpose_78/kernel
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
?0"
trackable_list_wrapper
�
@	variables
�metrics
Aregularization_losses
�layers
Btrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*�2batch_normalization_212/gamma
+:)�2batch_normalization_212/beta
4:2� (2#batch_normalization_212/moving_mean
8:6� (2'batch_normalization_212/moving_variance
<
E0
F1
G2
H3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
�
I	variables
�metrics
Jregularization_losses
�layers
Ktrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
M	variables
�metrics
Nregularization_losses
�layers
Otrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
5:3@�2conv2d_transpose_79/kernel
'
Q0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
�
R	variables
�metrics
Sregularization_losses
�layers
Ttrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)@2batch_normalization_213/gamma
*:(@2batch_normalization_213/beta
3:1@ (2#batch_normalization_213/moving_mean
7:5@ (2'batch_normalization_213/moving_variance
<
W0
X1
Y2
Z3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
�
[	variables
�metrics
\regularization_losses
�layers
]trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
_	variables
�metrics
`regularization_losses
�layers
atrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
4:2@2conv2d_transpose_80/kernel
'
c0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
c0"
trackable_list_wrapper
�
d	variables
�metrics
eregularization_losses
�layers
ftrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
	7

8
9
10
11
12
13"
trackable_list_wrapper
X
0
1
12
23
G4
H5
Y6
Z7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
.__inference_sequential_53_layer_call_fn_839815
.__inference_sequential_53_layer_call_fn_839843
.__inference_sequential_53_layer_call_fn_839240
.__inference_sequential_53_layer_call_fn_839171�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_838172�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
dense_101_input���������d
�2�
I__inference_sequential_53_layer_call_and_return_conditional_losses_839787
I__inference_sequential_53_layer_call_and_return_conditional_losses_839604
I__inference_sequential_53_layer_call_and_return_conditional_losses_839060
I__inference_sequential_53_layer_call_and_return_conditional_losses_839101�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dense_101_layer_call_fn_839860�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_101_layer_call_and_return_conditional_losses_839853�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_210_layer_call_fn_839967
8__inference_batch_normalization_210_layer_call_fn_839976�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_839958
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_839935�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_leaky_re_lu_207_layer_call_fn_839986�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_leaky_re_lu_207_layer_call_and_return_conditional_losses_839981�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_102_layer_call_fn_840003�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_102_layer_call_and_return_conditional_losses_839996�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_211_layer_call_fn_840119
8__inference_batch_normalization_211_layer_call_fn_840110�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_840101
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_840078�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_leaky_re_lu_208_layer_call_fn_840129�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_leaky_re_lu_208_layer_call_and_return_conditional_losses_840124�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_reshape_26_layer_call_fn_840148�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_reshape_26_layer_call_and_return_conditional_losses_840143�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_conv2d_transpose_78_layer_call_fn_838498�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
O__inference_conv2d_transpose_78_layer_call_and_return_conditional_losses_838491�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
8__inference_batch_normalization_212_layer_call_fn_840225
8__inference_batch_normalization_212_layer_call_fn_840234�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_840194
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_840216�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_leaky_re_lu_209_layer_call_fn_840244�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_leaky_re_lu_209_layer_call_and_return_conditional_losses_840239�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_conv2d_transpose_79_layer_call_fn_838668�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
O__inference_conv2d_transpose_79_layer_call_and_return_conditional_losses_838661�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
8__inference_batch_normalization_213_layer_call_fn_840321
8__inference_batch_normalization_213_layer_call_fn_840330�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_840312
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_840290�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_leaky_re_lu_210_layer_call_fn_840340�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_leaky_re_lu_210_layer_call_and_return_conditional_losses_840335�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_conv2d_transpose_80_layer_call_fn_838839�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
O__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_838832�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
;B9
$__inference_signature_wrapper_839365dense_101_input�
!__inference__wrapped_model_838172�()2/10?EFGHQWXYZc8�5
.�+
)�&
dense_101_input���������d
� "Q�N
L
conv2d_transpose_805�2
conv2d_transpose_80����������
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_839935d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
S__inference_batch_normalization_210_layer_call_and_return_conditional_losses_839958d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
8__inference_batch_normalization_210_layer_call_fn_839967W4�1
*�'
!�
inputs����������
p
� "������������
8__inference_batch_normalization_210_layer_call_fn_839976W4�1
*�'
!�
inputs����������
p 
� "������������
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_840078d12/04�1
*�'
!�
inputs����������b
p
� "&�#
�
0����������b
� �
S__inference_batch_normalization_211_layer_call_and_return_conditional_losses_840101d2/104�1
*�'
!�
inputs����������b
p 
� "&�#
�
0����������b
� �
8__inference_batch_normalization_211_layer_call_fn_840110W12/04�1
*�'
!�
inputs����������b
p
� "�����������b�
8__inference_batch_normalization_211_layer_call_fn_840119W2/104�1
*�'
!�
inputs����������b
p 
� "�����������b�
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_840194�EFGHN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_212_layer_call_and_return_conditional_losses_840216�EFGHN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
8__inference_batch_normalization_212_layer_call_fn_840225�EFGHN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_212_layer_call_fn_840234�EFGHN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_840290�WXYZM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
S__inference_batch_normalization_213_layer_call_and_return_conditional_losses_840312�WXYZM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
8__inference_batch_normalization_213_layer_call_fn_840321�WXYZM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
8__inference_batch_normalization_213_layer_call_fn_840330�WXYZM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
O__inference_conv2d_transpose_78_layer_call_and_return_conditional_losses_838491�?J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
4__inference_conv2d_transpose_78_layer_call_fn_838498�?J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
O__inference_conv2d_transpose_79_layer_call_and_return_conditional_losses_838661�QJ�G
@�=
;�8
inputs,����������������������������
� "?�<
5�2
0+���������������������������@
� �
4__inference_conv2d_transpose_79_layer_call_fn_838668�QJ�G
@�=
;�8
inputs,����������������������������
� "2�/+���������������������������@�
O__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_838832�cI�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������
� �
4__inference_conv2d_transpose_80_layer_call_fn_838839�cI�F
?�<
:�7
inputs+���������������������������@
� "2�/+����������������������������
E__inference_dense_101_layer_call_and_return_conditional_losses_839853]/�,
%�"
 �
inputs���������d
� "&�#
�
0����������
� ~
*__inference_dense_101_layer_call_fn_839860P/�,
%�"
 �
inputs���������d
� "������������
E__inference_dense_102_layer_call_and_return_conditional_losses_839996^()0�-
&�#
!�
inputs����������
� "&�#
�
0����������b
� 
*__inference_dense_102_layer_call_fn_840003Q()0�-
&�#
!�
inputs����������
� "�����������b�
K__inference_leaky_re_lu_207_layer_call_and_return_conditional_losses_839981Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
0__inference_leaky_re_lu_207_layer_call_fn_839986M0�-
&�#
!�
inputs����������
� "������������
K__inference_leaky_re_lu_208_layer_call_and_return_conditional_losses_840124Z0�-
&�#
!�
inputs����������b
� "&�#
�
0����������b
� �
0__inference_leaky_re_lu_208_layer_call_fn_840129M0�-
&�#
!�
inputs����������b
� "�����������b�
K__inference_leaky_re_lu_209_layer_call_and_return_conditional_losses_840239�J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
0__inference_leaky_re_lu_209_layer_call_fn_840244�J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
K__inference_leaky_re_lu_210_layer_call_and_return_conditional_losses_840335�I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
0__inference_leaky_re_lu_210_layer_call_fn_840340I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
F__inference_reshape_26_layer_call_and_return_conditional_losses_840143b0�-
&�#
!�
inputs����������b
� ".�+
$�!
0����������
� �
+__inference_reshape_26_layer_call_fn_840148U0�-
&�#
!�
inputs����������b
� "!������������
I__inference_sequential_53_layer_call_and_return_conditional_losses_839060�()12/0?EFGHQWXYZc@�=
6�3
)�&
dense_101_input���������d
p

 
� "?�<
5�2
0+���������������������������
� �
I__inference_sequential_53_layer_call_and_return_conditional_losses_839101�()2/10?EFGHQWXYZc@�=
6�3
)�&
dense_101_input���������d
p 

 
� "?�<
5�2
0+���������������������������
� �
I__inference_sequential_53_layer_call_and_return_conditional_losses_839604�()12/0?EFGHQWXYZc7�4
-�*
 �
inputs���������d
p

 
� "-�*
#� 
0���������
� �
I__inference_sequential_53_layer_call_and_return_conditional_losses_839787�()2/10?EFGHQWXYZc7�4
-�*
 �
inputs���������d
p 

 
� "-�*
#� 
0���������
� �
.__inference_sequential_53_layer_call_fn_839171�()12/0?EFGHQWXYZc@�=
6�3
)�&
dense_101_input���������d
p

 
� "2�/+����������������������������
.__inference_sequential_53_layer_call_fn_839240�()2/10?EFGHQWXYZc@�=
6�3
)�&
dense_101_input���������d
p 

 
� "2�/+����������������������������
.__inference_sequential_53_layer_call_fn_839815�()12/0?EFGHQWXYZc7�4
-�*
 �
inputs���������d
p

 
� "2�/+����������������������������
.__inference_sequential_53_layer_call_fn_839843�()2/10?EFGHQWXYZc7�4
-�*
 �
inputs���������d
p 

 
� "2�/+����������������������������
$__inference_signature_wrapper_839365�()2/10?EFGHQWXYZcK�H
� 
A�>
<
dense_101_input)�&
dense_101_input���������d"Q�N
L
conv2d_transpose_805�2
conv2d_transpose_80���������