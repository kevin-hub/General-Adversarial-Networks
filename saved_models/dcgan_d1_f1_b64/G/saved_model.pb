ад
ж¤
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
dtypetypeИ
╛
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02v2.1.0-0-ge5bf8de4108┘И
{
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dА* 
shared_namedense_33/kernel
t
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes
:	dА*
dtype0
s
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_33/bias
l
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes	
:А*
dtype0
С
batch_normalization_74/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_74/gamma
К
0batch_normalization_74/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_74/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_74/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_74/beta
И
/batch_normalization_74/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_74/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_74/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_74/moving_mean
Ц
6batch_normalization_74/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_74/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_74/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_74/moving_variance
Ю
:batch_normalization_74/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_74/moving_variance*
_output_shapes	
:А*
dtype0
|
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ААb* 
shared_namedense_34/kernel
u
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel* 
_output_shapes
:
ААb*
dtype0
s
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Аb*
shared_namedense_34/bias
l
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes	
:Аb*
dtype0
С
batch_normalization_75/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Аb*-
shared_namebatch_normalization_75/gamma
К
0batch_normalization_75/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_75/gamma*
_output_shapes	
:Аb*
dtype0
П
batch_normalization_75/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Аb*,
shared_namebatch_normalization_75/beta
И
/batch_normalization_75/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_75/beta*
_output_shapes	
:Аb*
dtype0
Э
"batch_normalization_75/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Аb*3
shared_name$"batch_normalization_75/moving_mean
Ц
6batch_normalization_75/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_75/moving_mean*
_output_shapes	
:Аb*
dtype0
е
&batch_normalization_75/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Аb*7
shared_name(&batch_normalization_75/moving_variance
Ю
:batch_normalization_75/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_75/moving_variance*
_output_shapes	
:Аb*
dtype0
Ъ
conv2d_transpose_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*+
shared_nameconv2d_transpose_27/kernel
У
.conv2d_transpose_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_27/kernel*(
_output_shapes
:АА*
dtype0
С
batch_normalization_76/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_76/gamma
К
0batch_normalization_76/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_76/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_76/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_76/beta
И
/batch_normalization_76/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_76/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_76/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_76/moving_mean
Ц
6batch_normalization_76/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_76/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_76/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_76/moving_variance
Ю
:batch_normalization_76/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_76/moving_variance*
_output_shapes	
:А*
dtype0
Щ
conv2d_transpose_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*+
shared_nameconv2d_transpose_28/kernel
Т
.conv2d_transpose_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_28/kernel*'
_output_shapes
:@А*
dtype0
Р
batch_normalization_77/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_77/gamma
Й
0batch_normalization_77/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_77/gamma*
_output_shapes
:@*
dtype0
О
batch_normalization_77/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_77/beta
З
/batch_normalization_77/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_77/beta*
_output_shapes
:@*
dtype0
Ь
"batch_normalization_77/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_77/moving_mean
Х
6batch_normalization_77/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_77/moving_mean*
_output_shapes
:@*
dtype0
д
&batch_normalization_77/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_77/moving_variance
Э
:batch_normalization_77/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_77/moving_variance*
_output_shapes
:@*
dtype0
Ш
conv2d_transpose_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameconv2d_transpose_29/kernel
С
.conv2d_transpose_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_29/kernel*&
_output_shapes
:@*
dtype0

NoOpNoOp
╬>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Й>
value =B№= Bї=
Ф
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
Ч
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
Ч
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
Ч
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
Ч
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
о
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
Ъ
	variables
hmetrics
trainable_variables

ilayers
regularization_losses
jnon_trainable_variables
klayer_regularization_losses
 
[Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Ъ
	variables
lmetrics
regularization_losses

mlayers
trainable_variables
nnon_trainable_variables
olayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_74/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_74/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_74/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_74/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

0
1
Ъ
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
Ъ
$	variables
tmetrics
%regularization_losses

ulayers
&trainable_variables
vnon_trainable_variables
wlayer_regularization_losses
[Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_34/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
Ъ
*	variables
xmetrics
+regularization_losses

ylayers
,trainable_variables
znon_trainable_variables
{layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_75/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_75/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_75/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_75/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

/0
01
12
23
 

/0
01
Ъ
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
Ю
7	variables
Аmetrics
8regularization_losses
Бlayers
9trainable_variables
Вnon_trainable_variables
 Гlayer_regularization_losses
 
 
 
Ю
;	variables
Дmetrics
<regularization_losses
Еlayers
=trainable_variables
Жnon_trainable_variables
 Зlayer_regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_27/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

?0
 

?0
Ю
@	variables
Иmetrics
Aregularization_losses
Йlayers
Btrainable_variables
Кnon_trainable_variables
 Лlayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_76/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_76/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_76/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_76/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

E0
F1
G2
H3
 

E0
F1
Ю
I	variables
Мmetrics
Jregularization_losses
Нlayers
Ktrainable_variables
Оnon_trainable_variables
 Пlayer_regularization_losses
 
 
 
Ю
M	variables
Рmetrics
Nregularization_losses
Сlayers
Otrainable_variables
Тnon_trainable_variables
 Уlayer_regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_28/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE

Q0
 

Q0
Ю
R	variables
Фmetrics
Sregularization_losses
Хlayers
Ttrainable_variables
Цnon_trainable_variables
 Чlayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_77/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_77/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_77/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_77/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
Y2
Z3
 

W0
X1
Ю
[	variables
Шmetrics
\regularization_losses
Щlayers
]trainable_variables
Ъnon_trainable_variables
 Ыlayer_regularization_losses
 
 
 
Ю
_	variables
Ьmetrics
`regularization_losses
Эlayers
atrainable_variables
Юnon_trainable_variables
 Яlayer_regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_29/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE

c0
 

c0
Ю
d	variables
аmetrics
eregularization_losses
бlayers
ftrainable_variables
вnon_trainable_variables
 гlayer_regularization_losses
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
Б
serving_default_dense_33_inputPlaceholder*'
_output_shapes
:         d*
dtype0*
shape:         d
є
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_33_inputdense_33/kerneldense_33/bias&batch_normalization_74/moving_variancebatch_normalization_74/gamma"batch_normalization_74/moving_meanbatch_normalization_74/betadense_34/kerneldense_34/bias&batch_normalization_75/moving_variancebatch_normalization_75/gamma"batch_normalization_75/moving_meanbatch_normalization_75/betaconv2d_transpose_27/kernelbatch_normalization_76/gammabatch_normalization_76/beta"batch_normalization_76/moving_mean&batch_normalization_76/moving_varianceconv2d_transpose_28/kernelbatch_normalization_77/gammabatch_normalization_77/beta"batch_normalization_77/moving_mean&batch_normalization_77/moving_varianceconv2d_transpose_29/kernel*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*-
f(R&
$__inference_signature_wrapper_375868
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ы
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOp0batch_normalization_74/gamma/Read/ReadVariableOp/batch_normalization_74/beta/Read/ReadVariableOp6batch_normalization_74/moving_mean/Read/ReadVariableOp:batch_normalization_74/moving_variance/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp0batch_normalization_75/gamma/Read/ReadVariableOp/batch_normalization_75/beta/Read/ReadVariableOp6batch_normalization_75/moving_mean/Read/ReadVariableOp:batch_normalization_75/moving_variance/Read/ReadVariableOp.conv2d_transpose_27/kernel/Read/ReadVariableOp0batch_normalization_76/gamma/Read/ReadVariableOp/batch_normalization_76/beta/Read/ReadVariableOp6batch_normalization_76/moving_mean/Read/ReadVariableOp:batch_normalization_76/moving_variance/Read/ReadVariableOp.conv2d_transpose_28/kernel/Read/ReadVariableOp0batch_normalization_77/gamma/Read/ReadVariableOp/batch_normalization_77/beta/Read/ReadVariableOp6batch_normalization_77/moving_mean/Read/ReadVariableOp:batch_normalization_77/moving_variance/Read/ReadVariableOp.conv2d_transpose_29/kernel/Read/ReadVariableOpConst*$
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
__inference__traced_save_376936
╩
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_33/kerneldense_33/biasbatch_normalization_74/gammabatch_normalization_74/beta"batch_normalization_74/moving_mean&batch_normalization_74/moving_variancedense_34/kerneldense_34/biasbatch_normalization_75/gammabatch_normalization_75/beta"batch_normalization_75/moving_mean&batch_normalization_75/moving_varianceconv2d_transpose_27/kernelbatch_normalization_76/gammabatch_normalization_76/beta"batch_normalization_76/moving_mean&batch_normalization_76/moving_varianceconv2d_transpose_28/kernelbatch_normalization_77/gammabatch_normalization_77/beta"batch_normalization_77/moving_mean&batch_normalization_77/moving_varianceconv2d_transpose_29/kernel*#
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
"__inference__traced_restore_377017А∙
С
С
4__inference_conv2d_transpose_28_layer_call_fn_375171

inputs"
statefulpartitionedcall_args_1
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3751642
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,                           А:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
┴
a
E__inference_reshape_9_layer_call_and_return_conditional_losses_375476

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
strided_slice/stack_2т
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
B :А2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:         А2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Аb:& "
 
_user_specified_nameinputs
ы
F
*__inference_reshape_9_layer_call_fn_376651

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_reshape_9_layer_call_and_return_conditional_losses_3754762
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Аb:& "
 
_user_specified_nameinputs
═
▒
.__inference_sequential_19_layer_call_fn_376318

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
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_3756482
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╗$
Э
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_375265

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_375250
assignmovingavg_1_375257
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/375250*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xп
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/375250*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_375250*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp╠
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/375250*
_output_shapes
:@2
AssignMovingAvg/sub_1╡
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/375250*
_output_shapes
:@2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_375250AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/375250*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/375257*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╖
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/375257*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_375257*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╪
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/375257*
_output_shapes
:@2
AssignMovingAvg_1/sub_1┐
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/375257*
_output_shapes
:@2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_375257AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/375257*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╕
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ї
А
7__inference_batch_normalization_76_layer_call_fn_376728

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_3750952
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Э
f
J__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_376484

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         А2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
ь
f
J__inference_leaky_re_lu_73_layer_call_and_return_conditional_losses_375514

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,                           А2
	LeakyReluЖ
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:& "
 
_user_specified_nameinputs
ШХ
д
I__inference_sequential_19_layer_call_and_return_conditional_losses_376290

inputs+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource<
8batch_normalization_74_batchnorm_readvariableop_resource@
<batch_normalization_74_batchnorm_mul_readvariableop_resource>
:batch_normalization_74_batchnorm_readvariableop_1_resource>
:batch_normalization_74_batchnorm_readvariableop_2_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource<
8batch_normalization_75_batchnorm_readvariableop_resource@
<batch_normalization_75_batchnorm_mul_readvariableop_resource>
:batch_normalization_75_batchnorm_readvariableop_1_resource>
:batch_normalization_75_batchnorm_readvariableop_2_resource@
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource2
.batch_normalization_76_readvariableop_resource4
0batch_normalization_76_readvariableop_1_resourceC
?batch_normalization_76_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource2
.batch_normalization_77_readvariableop_resource4
0batch_normalization_77_readvariableop_1_resourceC
?batch_normalization_77_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_29_conv2d_transpose_readvariableop_resource
identityИв/batch_normalization_74/batchnorm/ReadVariableOpв1batch_normalization_74/batchnorm/ReadVariableOp_1в1batch_normalization_74/batchnorm/ReadVariableOp_2в3batch_normalization_74/batchnorm/mul/ReadVariableOpв/batch_normalization_75/batchnorm/ReadVariableOpв1batch_normalization_75/batchnorm/ReadVariableOp_1в1batch_normalization_75/batchnorm/ReadVariableOp_2в3batch_normalization_75/batchnorm/mul/ReadVariableOpв6batch_normalization_76/FusedBatchNormV3/ReadVariableOpв8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_76/ReadVariableOpв'batch_normalization_76/ReadVariableOp_1в6batch_normalization_77/FusedBatchNormV3/ReadVariableOpв8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_77/ReadVariableOpв'batch_normalization_77/ReadVariableOp_1в3conv2d_transpose_27/conv2d_transpose/ReadVariableOpв3conv2d_transpose_28/conv2d_transpose/ReadVariableOpв3conv2d_transpose_29/conv2d_transpose/ReadVariableOpвdense_33/BiasAdd/ReadVariableOpвdense_33/MatMul/ReadVariableOpвdense_34/BiasAdd/ReadVariableOpвdense_34/MatMul/ReadVariableOpй
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	dА*
dtype02 
dense_33/MatMul/ReadVariableOpП
dense_33/MatMulMatMulinputs&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_33/MatMulи
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_33/BiasAdd/ReadVariableOpж
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_33/BiasAddМ
#batch_normalization_74/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_74/LogicalAnd/xМ
#batch_normalization_74/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_74/LogicalAnd/y╚
!batch_normalization_74/LogicalAnd
LogicalAnd,batch_normalization_74/LogicalAnd/x:output:0,batch_normalization_74/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_74/LogicalAnd╪
/batch_normalization_74/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_74_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/batch_normalization_74/batchnorm/ReadVariableOpХ
&batch_normalization_74/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2(
&batch_normalization_74/batchnorm/add/yх
$batch_normalization_74/batchnorm/addAddV27batch_normalization_74/batchnorm/ReadVariableOp:value:0/batch_normalization_74/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2&
$batch_normalization_74/batchnorm/addй
&batch_normalization_74/batchnorm/RsqrtRsqrt(batch_normalization_74/batchnorm/add:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_74/batchnorm/Rsqrtф
3batch_normalization_74/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_74_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3batch_normalization_74/batchnorm/mul/ReadVariableOpт
$batch_normalization_74/batchnorm/mulMul*batch_normalization_74/batchnorm/Rsqrt:y:0;batch_normalization_74/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2&
$batch_normalization_74/batchnorm/mul╧
&batch_normalization_74/batchnorm/mul_1Muldense_33/BiasAdd:output:0(batch_normalization_74/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_74/batchnorm/mul_1▐
1batch_normalization_74/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_74_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype023
1batch_normalization_74/batchnorm/ReadVariableOp_1т
&batch_normalization_74/batchnorm/mul_2Mul9batch_normalization_74/batchnorm/ReadVariableOp_1:value:0(batch_normalization_74/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_74/batchnorm/mul_2▐
1batch_normalization_74/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_74_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype023
1batch_normalization_74/batchnorm/ReadVariableOp_2р
$batch_normalization_74/batchnorm/subSub9batch_normalization_74/batchnorm/ReadVariableOp_2:value:0*batch_normalization_74/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2&
$batch_normalization_74/batchnorm/subт
&batch_normalization_74/batchnorm/add_1AddV2*batch_normalization_74/batchnorm/mul_1:z:0(batch_normalization_74/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_74/batchnorm/add_1Ч
leaky_re_lu_71/LeakyRelu	LeakyRelu*batch_normalization_74/batchnorm/add_1:z:0*(
_output_shapes
:         А2
leaky_re_lu_71/LeakyReluк
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource* 
_output_shapes
:
ААb*
dtype02 
dense_34/MatMul/ReadVariableOpп
dense_34/MatMulMatMul&leaky_re_lu_71/LeakyRelu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аb2
dense_34/MatMulи
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes	
:Аb*
dtype02!
dense_34/BiasAdd/ReadVariableOpж
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аb2
dense_34/BiasAddМ
#batch_normalization_75/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_75/LogicalAnd/xМ
#batch_normalization_75/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_75/LogicalAnd/y╚
!batch_normalization_75/LogicalAnd
LogicalAnd,batch_normalization_75/LogicalAnd/x:output:0,batch_normalization_75/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_75/LogicalAnd╪
/batch_normalization_75/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_75_batchnorm_readvariableop_resource*
_output_shapes	
:Аb*
dtype021
/batch_normalization_75/batchnorm/ReadVariableOpХ
&batch_normalization_75/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2(
&batch_normalization_75/batchnorm/add/yх
$batch_normalization_75/batchnorm/addAddV27batch_normalization_75/batchnorm/ReadVariableOp:value:0/batch_normalization_75/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Аb2&
$batch_normalization_75/batchnorm/addй
&batch_normalization_75/batchnorm/RsqrtRsqrt(batch_normalization_75/batchnorm/add:z:0*
T0*
_output_shapes	
:Аb2(
&batch_normalization_75/batchnorm/Rsqrtф
3batch_normalization_75/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_75_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Аb*
dtype025
3batch_normalization_75/batchnorm/mul/ReadVariableOpт
$batch_normalization_75/batchnorm/mulMul*batch_normalization_75/batchnorm/Rsqrt:y:0;batch_normalization_75/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аb2&
$batch_normalization_75/batchnorm/mul╧
&batch_normalization_75/batchnorm/mul_1Muldense_34/BiasAdd:output:0(batch_normalization_75/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Аb2(
&batch_normalization_75/batchnorm/mul_1▐
1batch_normalization_75/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_75_batchnorm_readvariableop_1_resource*
_output_shapes	
:Аb*
dtype023
1batch_normalization_75/batchnorm/ReadVariableOp_1т
&batch_normalization_75/batchnorm/mul_2Mul9batch_normalization_75/batchnorm/ReadVariableOp_1:value:0(batch_normalization_75/batchnorm/mul:z:0*
T0*
_output_shapes	
:Аb2(
&batch_normalization_75/batchnorm/mul_2▐
1batch_normalization_75/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_75_batchnorm_readvariableop_2_resource*
_output_shapes	
:Аb*
dtype023
1batch_normalization_75/batchnorm/ReadVariableOp_2р
$batch_normalization_75/batchnorm/subSub9batch_normalization_75/batchnorm/ReadVariableOp_2:value:0*batch_normalization_75/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аb2&
$batch_normalization_75/batchnorm/subт
&batch_normalization_75/batchnorm/add_1AddV2*batch_normalization_75/batchnorm/mul_1:z:0(batch_normalization_75/batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аb2(
&batch_normalization_75/batchnorm/add_1Ч
leaky_re_lu_72/LeakyRelu	LeakyRelu*batch_normalization_75/batchnorm/add_1:z:0*(
_output_shapes
:         Аb2
leaky_re_lu_72/LeakyRelux
reshape_9/ShapeShape&leaky_re_lu_72/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_9/ShapeИ
reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_9/strided_slice/stackМ
reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_1М
reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_2Ю
reshape_9/strided_sliceStridedSlicereshape_9/Shape:output:0&reshape_9/strided_slice/stack:output:0(reshape_9/strided_slice/stack_1:output:0(reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_9/strided_slicex
reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/1x
reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/2y
reshape_9/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :А2
reshape_9/Reshape/shape/3Ў
reshape_9/Reshape/shapePack reshape_9/strided_slice:output:0"reshape_9/Reshape/shape/1:output:0"reshape_9/Reshape/shape/2:output:0"reshape_9/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_9/Reshape/shape╢
reshape_9/ReshapeReshape&leaky_re_lu_72/LeakyRelu:activations:0 reshape_9/Reshape/shape:output:0*
T0*0
_output_shapes
:         А2
reshape_9/ReshapeА
conv2d_transpose_27/ShapeShapereshape_9/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_27/ShapeЬ
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_27/strided_slice/stackа
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice/stack_1а
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice/stack_2┌
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_27/strided_sliceа
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice_1/stackд
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_1/stack_1д
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_1/stack_2ф
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/Shape:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_27/strided_slice_1а
)conv2d_transpose_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice_2/stackд
+conv2d_transpose_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_2/stack_1д
+conv2d_transpose_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_2/stack_2ф
#conv2d_transpose_27/strided_slice_2StridedSlice"conv2d_transpose_27/Shape:output:02conv2d_transpose_27/strided_slice_2/stack:output:04conv2d_transpose_27/strided_slice_2/stack_1:output:04conv2d_transpose_27/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_27/strided_slice_2x
conv2d_transpose_27/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_27/mul/yм
conv2d_transpose_27/mulMul,conv2d_transpose_27/strided_slice_1:output:0"conv2d_transpose_27/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_27/mul|
conv2d_transpose_27/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_27/mul_1/y▓
conv2d_transpose_27/mul_1Mul,conv2d_transpose_27/strided_slice_2:output:0$conv2d_transpose_27/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_27/mul_1}
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2
conv2d_transpose_27/stack/3·
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0conv2d_transpose_27/mul:z:0conv2d_transpose_27/mul_1:z:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_27/stackа
)conv2d_transpose_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_27/strided_slice_3/stackд
+conv2d_transpose_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_3/stack_1д
+conv2d_transpose_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_3/stack_2ф
#conv2d_transpose_27/strided_slice_3StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_3/stack:output:04conv2d_transpose_27/strided_slice_3/stack_1:output:04conv2d_transpose_27/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_27/strided_slice_3ё
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype025
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp├
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0reshape_9/Reshape:output:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2&
$conv2d_transpose_27/conv2d_transposeМ
#batch_normalization_76/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_76/LogicalAnd/xМ
#batch_normalization_76/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_76/LogicalAnd/y╚
!batch_normalization_76/LogicalAnd
LogicalAnd,batch_normalization_76/LogicalAnd/x:output:0,batch_normalization_76/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_76/LogicalAnd║
%batch_normalization_76/ReadVariableOpReadVariableOp.batch_normalization_76_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_76/ReadVariableOp└
'batch_normalization_76/ReadVariableOp_1ReadVariableOp0batch_normalization_76_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_76/ReadVariableOp_1э
6batch_normalization_76/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_76_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_76/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1А
'batch_normalization_76/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_27/conv2d_transpose:output:0-batch_normalization_76/ReadVariableOp:value:0/batch_normalization_76/ReadVariableOp_1:value:0>batch_normalization_76/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_76/FusedBatchNormV3Б
batch_normalization_76/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
batch_normalization_76/Constа
leaky_re_lu_73/LeakyRelu	LeakyRelu+batch_normalization_76/FusedBatchNormV3:y:0*0
_output_shapes
:         А2
leaky_re_lu_73/LeakyReluМ
conv2d_transpose_28/ShapeShape&leaky_re_lu_73/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_28/ShapeЬ
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_28/strided_slice/stackа
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice/stack_1а
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice/stack_2┌
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_28/strided_sliceа
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice_1/stackд
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_1/stack_1д
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_1/stack_2ф
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/Shape:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_28/strided_slice_1а
)conv2d_transpose_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice_2/stackд
+conv2d_transpose_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_2/stack_1д
+conv2d_transpose_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_2/stack_2ф
#conv2d_transpose_28/strided_slice_2StridedSlice"conv2d_transpose_28/Shape:output:02conv2d_transpose_28/strided_slice_2/stack:output:04conv2d_transpose_28/strided_slice_2/stack_1:output:04conv2d_transpose_28/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_28/strided_slice_2x
conv2d_transpose_28/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_28/mul/yм
conv2d_transpose_28/mulMul,conv2d_transpose_28/strided_slice_1:output:0"conv2d_transpose_28/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_28/mul|
conv2d_transpose_28/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_28/mul_1/y▓
conv2d_transpose_28/mul_1Mul,conv2d_transpose_28/strided_slice_2:output:0$conv2d_transpose_28/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_28/mul_1|
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_28/stack/3·
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0conv2d_transpose_28/mul:z:0conv2d_transpose_28/mul_1:z:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_28/stackа
)conv2d_transpose_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_28/strided_slice_3/stackд
+conv2d_transpose_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_3/stack_1д
+conv2d_transpose_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_3/stack_2ф
#conv2d_transpose_28/strided_slice_3StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_3/stack:output:04conv2d_transpose_28/strided_slice_3/stack_1:output:04conv2d_transpose_28/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_28/strided_slice_3Ё
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype025
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp╬
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_73/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2&
$conv2d_transpose_28/conv2d_transposeМ
#batch_normalization_77/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_77/LogicalAnd/xМ
#batch_normalization_77/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_77/LogicalAnd/y╚
!batch_normalization_77/LogicalAnd
LogicalAnd,batch_normalization_77/LogicalAnd/x:output:0,batch_normalization_77/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_77/LogicalAnd╣
%batch_normalization_77/ReadVariableOpReadVariableOp.batch_normalization_77_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_77/ReadVariableOp┐
'batch_normalization_77/ReadVariableOp_1ReadVariableOp0batch_normalization_77_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_77/ReadVariableOp_1ь
6batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_77/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1√
'batch_normalization_77/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_28/conv2d_transpose:output:0-batch_normalization_77/ReadVariableOp:value:0/batch_normalization_77/ReadVariableOp_1:value:0>batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_77/FusedBatchNormV3Б
batch_normalization_77/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
batch_normalization_77/ConstЯ
leaky_re_lu_74/LeakyRelu	LeakyRelu+batch_normalization_77/FusedBatchNormV3:y:0*/
_output_shapes
:         @2
leaky_re_lu_74/LeakyReluМ
conv2d_transpose_29/ShapeShape&leaky_re_lu_74/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_29/ShapeЬ
'conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_29/strided_slice/stackа
)conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_29/strided_slice/stack_1а
)conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_29/strided_slice/stack_2┌
!conv2d_transpose_29/strided_sliceStridedSlice"conv2d_transpose_29/Shape:output:00conv2d_transpose_29/strided_slice/stack:output:02conv2d_transpose_29/strided_slice/stack_1:output:02conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_29/strided_sliceа
)conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_29/strided_slice_1/stackд
+conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_1/stack_1д
+conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_1/stack_2ф
#conv2d_transpose_29/strided_slice_1StridedSlice"conv2d_transpose_29/Shape:output:02conv2d_transpose_29/strided_slice_1/stack:output:04conv2d_transpose_29/strided_slice_1/stack_1:output:04conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_29/strided_slice_1а
)conv2d_transpose_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_29/strided_slice_2/stackд
+conv2d_transpose_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_2/stack_1д
+conv2d_transpose_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_2/stack_2ф
#conv2d_transpose_29/strided_slice_2StridedSlice"conv2d_transpose_29/Shape:output:02conv2d_transpose_29/strided_slice_2/stack:output:04conv2d_transpose_29/strided_slice_2/stack_1:output:04conv2d_transpose_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_29/strided_slice_2x
conv2d_transpose_29/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_29/mul/yм
conv2d_transpose_29/mulMul,conv2d_transpose_29/strided_slice_1:output:0"conv2d_transpose_29/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_29/mul|
conv2d_transpose_29/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_29/mul_1/y▓
conv2d_transpose_29/mul_1Mul,conv2d_transpose_29/strided_slice_2:output:0$conv2d_transpose_29/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_29/mul_1|
conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_29/stack/3·
conv2d_transpose_29/stackPack*conv2d_transpose_29/strided_slice:output:0conv2d_transpose_29/mul:z:0conv2d_transpose_29/mul_1:z:0$conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_29/stackа
)conv2d_transpose_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_29/strided_slice_3/stackд
+conv2d_transpose_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_3/stack_1д
+conv2d_transpose_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_3/stack_2ф
#conv2d_transpose_29/strided_slice_3StridedSlice"conv2d_transpose_29/stack:output:02conv2d_transpose_29/strided_slice_3/stack:output:04conv2d_transpose_29/strided_slice_3/stack_1:output:04conv2d_transpose_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_29/strided_slice_3я
3conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype025
3conv2d_transpose_29/conv2d_transpose/ReadVariableOp╬
$conv2d_transpose_29/conv2d_transposeConv2DBackpropInput"conv2d_transpose_29/stack:output:0;conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_74/LeakyRelu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2&
$conv2d_transpose_29/conv2d_transposeе
conv2d_transpose_29/TanhTanh-conv2d_transpose_29/conv2d_transpose:output:0*
T0*/
_output_shapes
:         2
conv2d_transpose_29/Tanh╠	
IdentityIdentityconv2d_transpose_29/Tanh:y:00^batch_normalization_74/batchnorm/ReadVariableOp2^batch_normalization_74/batchnorm/ReadVariableOp_12^batch_normalization_74/batchnorm/ReadVariableOp_24^batch_normalization_74/batchnorm/mul/ReadVariableOp0^batch_normalization_75/batchnorm/ReadVariableOp2^batch_normalization_75/batchnorm/ReadVariableOp_12^batch_normalization_75/batchnorm/ReadVariableOp_24^batch_normalization_75/batchnorm/mul/ReadVariableOp7^batch_normalization_76/FusedBatchNormV3/ReadVariableOp9^batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_76/ReadVariableOp(^batch_normalization_76/ReadVariableOp_17^batch_normalization_77/FusedBatchNormV3/ReadVariableOp9^batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_77/ReadVariableOp(^batch_normalization_77/ReadVariableOp_14^conv2d_transpose_27/conv2d_transpose/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp4^conv2d_transpose_29/conv2d_transpose/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::2b
/batch_normalization_74/batchnorm/ReadVariableOp/batch_normalization_74/batchnorm/ReadVariableOp2f
1batch_normalization_74/batchnorm/ReadVariableOp_11batch_normalization_74/batchnorm/ReadVariableOp_12f
1batch_normalization_74/batchnorm/ReadVariableOp_21batch_normalization_74/batchnorm/ReadVariableOp_22j
3batch_normalization_74/batchnorm/mul/ReadVariableOp3batch_normalization_74/batchnorm/mul/ReadVariableOp2b
/batch_normalization_75/batchnorm/ReadVariableOp/batch_normalization_75/batchnorm/ReadVariableOp2f
1batch_normalization_75/batchnorm/ReadVariableOp_11batch_normalization_75/batchnorm/ReadVariableOp_12f
1batch_normalization_75/batchnorm/ReadVariableOp_21batch_normalization_75/batchnorm/ReadVariableOp_22j
3batch_normalization_75/batchnorm/mul/ReadVariableOp3batch_normalization_75/batchnorm/mul/ReadVariableOp2p
6batch_normalization_76/FusedBatchNormV3/ReadVariableOp6batch_normalization_76/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_18batch_normalization_76/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_76/ReadVariableOp%batch_normalization_76/ReadVariableOp2R
'batch_normalization_76/ReadVariableOp_1'batch_normalization_76/ReadVariableOp_12p
6batch_normalization_77/FusedBatchNormV3/ReadVariableOp6batch_normalization_77/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_18batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_77/ReadVariableOp%batch_normalization_77/ReadVariableOp2R
'batch_normalization_77/ReadVariableOp_1'batch_normalization_77/ReadVariableOp_12j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_29/conv2d_transpose/ReadVariableOp3conv2d_transpose_29/conv2d_transpose/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Э
f
J__inference_leaky_re_lu_72_layer_call_and_return_conditional_losses_376627

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         Аb2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Аb:& "
 
_user_specified_nameinputs
ў
к
)__inference_dense_33_layer_call_fn_376363

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_3753562
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
У
С
4__inference_conv2d_transpose_27_layer_call_fn_375001

inputs"
statefulpartitionedcall_args_1
identityИвStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3749942
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,                           А:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
┤/
╔
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_376581

inputs
assignmovingavg_376556
assignmovingavg_1_376562)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOp^
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

LogicalAndК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Аb*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Аb2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Аb2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Аb*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Аb*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Аb*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/376556*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_376556*
_output_shapes	
:Аb*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/376556*
_output_shapes	
:Аb2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/376556*
_output_shapes	
:Аb2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_376556AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/376556*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/376562*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_376562*
_output_shapes	
:Аb*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/376562*
_output_shapes	
:Аb2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/376562*
_output_shapes	
:Аb2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_376562AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/376562*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Аb2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аb2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Аb2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аb2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         Аb::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ё
▌
D__inference_dense_34_layer_call_and_return_conditional_losses_376499

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ААb*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аb2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Аb*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аb2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ж
А
7__inference_batch_normalization_74_layer_call_fn_376479

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_3748122
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Є
╛
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_375164

inputs,
(conv2d_transpose_readvariableop_resource
identityИвconv2d_transpose/ReadVariableOpD
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
strided_slice/stack_2т
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
strided_slice_1/stack_2ь
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
strided_slice_2/stack_2ь
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
stack/3В
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
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3┤
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype02!
conv2d_transpose/ReadVariableOpЁ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
conv2d_transposeй
IdentityIdentityconv2d_transpose:output:0 ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,                           А:2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
ч
Й
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_376461

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOp^
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

LogicalAndУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
х
K
/__inference_leaky_re_lu_72_layer_call_fn_376632

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_72_layer_call_and_return_conditional_losses_3754542
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Аb:& "
 
_user_specified_nameinputs
ё
А
7__inference_batch_normalization_77_layer_call_fn_376833

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_3752962
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
┤/
╔
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_374924

inputs
assignmovingavg_374899
assignmovingavg_1_374905)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOp^
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

LogicalAndК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Аb*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Аb2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Аb2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Аb*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Аb*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Аb*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/374899*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_374899*
_output_shapes	
:Аb*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/374899*
_output_shapes	
:Аb2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/374899*
_output_shapes	
:Аb2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_374899AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/374899*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/374905*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_374905*
_output_shapes	
:Аb*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/374905*
_output_shapes	
:Аb2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/374905*
_output_shapes	
:Аb2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_374905AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/374905*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Аb2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аb2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Аb2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аb2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         Аb::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
П
п
$__inference_signature_wrapper_375868
dense_33_input"
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
identityИвStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCalldense_33_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__wrapped_model_3746752
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namedense_33_input
Э
f
J__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_375396

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         А2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╙8
√
__inference__traced_save_376936
file_prefix.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop;
7savev2_batch_normalization_74_gamma_read_readvariableop:
6savev2_batch_normalization_74_beta_read_readvariableopA
=savev2_batch_normalization_74_moving_mean_read_readvariableopE
Asavev2_batch_normalization_74_moving_variance_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop;
7savev2_batch_normalization_75_gamma_read_readvariableop:
6savev2_batch_normalization_75_beta_read_readvariableopA
=savev2_batch_normalization_75_moving_mean_read_readvariableopE
Asavev2_batch_normalization_75_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_27_kernel_read_readvariableop;
7savev2_batch_normalization_76_gamma_read_readvariableop:
6savev2_batch_normalization_76_beta_read_readvariableopA
=savev2_batch_normalization_76_moving_mean_read_readvariableopE
Asavev2_batch_normalization_76_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_28_kernel_read_readvariableop;
7savev2_batch_normalization_77_gamma_read_readvariableop:
6savev2_batch_normalization_77_beta_read_readvariableopA
=savev2_batch_normalization_77_moving_mean_read_readvariableopE
Asavev2_batch_normalization_77_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_29_kernel_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1е
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_15befeadf11b4d7d83dcc8744bd55f29/part2
StringJoin/inputs_1Б

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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename│
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*┼

value╗
B╕
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names╢
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices▀
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop7savev2_batch_normalization_74_gamma_read_readvariableop6savev2_batch_normalization_74_beta_read_readvariableop=savev2_batch_normalization_74_moving_mean_read_readvariableopAsavev2_batch_normalization_74_moving_variance_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop7savev2_batch_normalization_75_gamma_read_readvariableop6savev2_batch_normalization_75_beta_read_readvariableop=savev2_batch_normalization_75_moving_mean_read_readvariableopAsavev2_batch_normalization_75_moving_variance_read_readvariableop5savev2_conv2d_transpose_27_kernel_read_readvariableop7savev2_batch_normalization_76_gamma_read_readvariableop6savev2_batch_normalization_76_beta_read_readvariableop=savev2_batch_normalization_76_moving_mean_read_readvariableopAsavev2_batch_normalization_76_moving_variance_read_readvariableop5savev2_conv2d_transpose_28_kernel_read_readvariableop7savev2_batch_normalization_77_gamma_read_readvariableop6savev2_batch_normalization_77_beta_read_readvariableop=savev2_batch_normalization_77_moving_mean_read_readvariableopAsavev2_batch_normalization_77_moving_variance_read_readvariableop5savev2_conv2d_transpose_29_kernel_read_readvariableop"/device:CPU:0*
_output_shapes
 *%
dtypes
22
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*у
_input_shapes╤
╬: :	dА:А:А:А:А:А:
ААb:Аb:Аb:Аb:Аb:Аb:АА:А:А:А:А:@А:@:@:@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
ж
А
7__inference_batch_normalization_75_layer_call_fn_376622

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_3749562
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         Аb::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ю
▌
D__inference_dense_33_layer_call_and_return_conditional_losses_376356

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╦Q
Р
I__inference_sequential_19_layer_call_and_return_conditional_losses_375604
dense_33_input+
'dense_33_statefulpartitionedcall_args_1+
'dense_33_statefulpartitionedcall_args_29
5batch_normalization_74_statefulpartitionedcall_args_19
5batch_normalization_74_statefulpartitionedcall_args_29
5batch_normalization_74_statefulpartitionedcall_args_39
5batch_normalization_74_statefulpartitionedcall_args_4+
'dense_34_statefulpartitionedcall_args_1+
'dense_34_statefulpartitionedcall_args_29
5batch_normalization_75_statefulpartitionedcall_args_19
5batch_normalization_75_statefulpartitionedcall_args_29
5batch_normalization_75_statefulpartitionedcall_args_39
5batch_normalization_75_statefulpartitionedcall_args_46
2conv2d_transpose_27_statefulpartitionedcall_args_19
5batch_normalization_76_statefulpartitionedcall_args_19
5batch_normalization_76_statefulpartitionedcall_args_29
5batch_normalization_76_statefulpartitionedcall_args_39
5batch_normalization_76_statefulpartitionedcall_args_46
2conv2d_transpose_28_statefulpartitionedcall_args_19
5batch_normalization_77_statefulpartitionedcall_args_19
5batch_normalization_77_statefulpartitionedcall_args_29
5batch_normalization_77_statefulpartitionedcall_args_39
5batch_normalization_77_statefulpartitionedcall_args_46
2conv2d_transpose_29_statefulpartitionedcall_args_1
identityИв.batch_normalization_74/StatefulPartitionedCallв.batch_normalization_75/StatefulPartitionedCallв.batch_normalization_76/StatefulPartitionedCallв.batch_normalization_77/StatefulPartitionedCallв+conv2d_transpose_27/StatefulPartitionedCallв+conv2d_transpose_28/StatefulPartitionedCallв+conv2d_transpose_29/StatefulPartitionedCallв dense_33/StatefulPartitionedCallв dense_34/StatefulPartitionedCall╢
 dense_33/StatefulPartitionedCallStatefulPartitionedCalldense_33_input'dense_33_statefulpartitionedcall_args_1'dense_33_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_3753562"
 dense_33/StatefulPartitionedCallЗ
.batch_normalization_74/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:05batch_normalization_74_statefulpartitionedcall_args_15batch_normalization_74_statefulpartitionedcall_args_25batch_normalization_74_statefulpartitionedcall_args_35batch_normalization_74_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_37481220
.batch_normalization_74/StatefulPartitionedCallЕ
leaky_re_lu_71/PartitionedCallPartitionedCall7batch_normalization_74/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_3753962 
leaky_re_lu_71/PartitionedCall╧
 dense_34/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0'dense_34_statefulpartitionedcall_args_1'dense_34_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3754142"
 dense_34/StatefulPartitionedCallЗ
.batch_normalization_75/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:05batch_normalization_75_statefulpartitionedcall_args_15batch_normalization_75_statefulpartitionedcall_args_25batch_normalization_75_statefulpartitionedcall_args_35batch_normalization_75_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_37495620
.batch_normalization_75/StatefulPartitionedCallЕ
leaky_re_lu_72/PartitionedCallPartitionedCall7batch_normalization_75/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_72_layer_call_and_return_conditional_losses_3754542 
leaky_re_lu_72/PartitionedCallю
reshape_9/PartitionedCallPartitionedCall'leaky_re_lu_72/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_reshape_9_layer_call_and_return_conditional_losses_3754762
reshape_9/PartitionedCallц
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall"reshape_9/PartitionedCall:output:02conv2d_transpose_27_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3749942-
+conv2d_transpose_27/StatefulPartitionedCallм
.batch_normalization_76/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:05batch_normalization_76_statefulpartitionedcall_args_15batch_normalization_76_statefulpartitionedcall_args_25batch_normalization_76_statefulpartitionedcall_args_35batch_normalization_76_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_37512620
.batch_normalization_76/StatefulPartitionedCallЯ
leaky_re_lu_73/PartitionedCallPartitionedCall7batch_normalization_76/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_73_layer_call_and_return_conditional_losses_3755142 
leaky_re_lu_73/PartitionedCallъ
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_73/PartitionedCall:output:02conv2d_transpose_28_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3751642-
+conv2d_transpose_28/StatefulPartitionedCallл
.batch_normalization_77/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:05batch_normalization_77_statefulpartitionedcall_args_15batch_normalization_77_statefulpartitionedcall_args_25batch_normalization_77_statefulpartitionedcall_args_35batch_normalization_77_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_37529620
.batch_normalization_77/StatefulPartitionedCallЮ
leaky_re_lu_74/PartitionedCallPartitionedCall7batch_normalization_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_74_layer_call_and_return_conditional_losses_3755522 
leaky_re_lu_74/PartitionedCallъ
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_74/PartitionedCall:output:02conv2d_transpose_29_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3753352-
+conv2d_transpose_29/StatefulPartitionedCall╢
IdentityIdentity4conv2d_transpose_29/StatefulPartitionedCall:output:0/^batch_normalization_74/StatefulPartitionedCall/^batch_normalization_75/StatefulPartitionedCall/^batch_normalization_76/StatefulPartitionedCall/^batch_normalization_77/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::2`
.batch_normalization_74/StatefulPartitionedCall.batch_normalization_74/StatefulPartitionedCall2`
.batch_normalization_75/StatefulPartitionedCall.batch_normalization_75/StatefulPartitionedCall2`
.batch_normalization_76/StatefulPartitionedCall.batch_normalization_76/StatefulPartitionedCall2`
.batch_normalization_77/StatefulPartitionedCall.batch_normalization_77/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:. *
(
_user_specified_namedense_33_input
х
╣
.__inference_sequential_19_layer_call_fn_375743
dense_33_input"
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
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCalldense_33_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_3757172
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namedense_33_input
Э
f
J__inference_leaky_re_lu_72_layer_call_and_return_conditional_losses_375454

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         Аb2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Аb:& "
 
_user_specified_nameinputs
│Q
И
I__inference_sequential_19_layer_call_and_return_conditional_losses_375717

inputs+
'dense_33_statefulpartitionedcall_args_1+
'dense_33_statefulpartitionedcall_args_29
5batch_normalization_74_statefulpartitionedcall_args_19
5batch_normalization_74_statefulpartitionedcall_args_29
5batch_normalization_74_statefulpartitionedcall_args_39
5batch_normalization_74_statefulpartitionedcall_args_4+
'dense_34_statefulpartitionedcall_args_1+
'dense_34_statefulpartitionedcall_args_29
5batch_normalization_75_statefulpartitionedcall_args_19
5batch_normalization_75_statefulpartitionedcall_args_29
5batch_normalization_75_statefulpartitionedcall_args_39
5batch_normalization_75_statefulpartitionedcall_args_46
2conv2d_transpose_27_statefulpartitionedcall_args_19
5batch_normalization_76_statefulpartitionedcall_args_19
5batch_normalization_76_statefulpartitionedcall_args_29
5batch_normalization_76_statefulpartitionedcall_args_39
5batch_normalization_76_statefulpartitionedcall_args_46
2conv2d_transpose_28_statefulpartitionedcall_args_19
5batch_normalization_77_statefulpartitionedcall_args_19
5batch_normalization_77_statefulpartitionedcall_args_29
5batch_normalization_77_statefulpartitionedcall_args_39
5batch_normalization_77_statefulpartitionedcall_args_46
2conv2d_transpose_29_statefulpartitionedcall_args_1
identityИв.batch_normalization_74/StatefulPartitionedCallв.batch_normalization_75/StatefulPartitionedCallв.batch_normalization_76/StatefulPartitionedCallв.batch_normalization_77/StatefulPartitionedCallв+conv2d_transpose_27/StatefulPartitionedCallв+conv2d_transpose_28/StatefulPartitionedCallв+conv2d_transpose_29/StatefulPartitionedCallв dense_33/StatefulPartitionedCallв dense_34/StatefulPartitionedCallо
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinputs'dense_33_statefulpartitionedcall_args_1'dense_33_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_3753562"
 dense_33/StatefulPartitionedCallЗ
.batch_normalization_74/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:05batch_normalization_74_statefulpartitionedcall_args_15batch_normalization_74_statefulpartitionedcall_args_25batch_normalization_74_statefulpartitionedcall_args_35batch_normalization_74_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_37481220
.batch_normalization_74/StatefulPartitionedCallЕ
leaky_re_lu_71/PartitionedCallPartitionedCall7batch_normalization_74/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_3753962 
leaky_re_lu_71/PartitionedCall╧
 dense_34/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0'dense_34_statefulpartitionedcall_args_1'dense_34_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3754142"
 dense_34/StatefulPartitionedCallЗ
.batch_normalization_75/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:05batch_normalization_75_statefulpartitionedcall_args_15batch_normalization_75_statefulpartitionedcall_args_25batch_normalization_75_statefulpartitionedcall_args_35batch_normalization_75_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_37495620
.batch_normalization_75/StatefulPartitionedCallЕ
leaky_re_lu_72/PartitionedCallPartitionedCall7batch_normalization_75/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_72_layer_call_and_return_conditional_losses_3754542 
leaky_re_lu_72/PartitionedCallю
reshape_9/PartitionedCallPartitionedCall'leaky_re_lu_72/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_reshape_9_layer_call_and_return_conditional_losses_3754762
reshape_9/PartitionedCallц
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall"reshape_9/PartitionedCall:output:02conv2d_transpose_27_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3749942-
+conv2d_transpose_27/StatefulPartitionedCallм
.batch_normalization_76/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:05batch_normalization_76_statefulpartitionedcall_args_15batch_normalization_76_statefulpartitionedcall_args_25batch_normalization_76_statefulpartitionedcall_args_35batch_normalization_76_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_37512620
.batch_normalization_76/StatefulPartitionedCallЯ
leaky_re_lu_73/PartitionedCallPartitionedCall7batch_normalization_76/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_73_layer_call_and_return_conditional_losses_3755142 
leaky_re_lu_73/PartitionedCallъ
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_73/PartitionedCall:output:02conv2d_transpose_28_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3751642-
+conv2d_transpose_28/StatefulPartitionedCallл
.batch_normalization_77/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:05batch_normalization_77_statefulpartitionedcall_args_15batch_normalization_77_statefulpartitionedcall_args_25batch_normalization_77_statefulpartitionedcall_args_35batch_normalization_77_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_37529620
.batch_normalization_77/StatefulPartitionedCallЮ
leaky_re_lu_74/PartitionedCallPartitionedCall7batch_normalization_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_74_layer_call_and_return_conditional_losses_3755522 
leaky_re_lu_74/PartitionedCallъ
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_74/PartitionedCall:output:02conv2d_transpose_29_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3753352-
+conv2d_transpose_29/StatefulPartitionedCall╢
IdentityIdentity4conv2d_transpose_29/StatefulPartitionedCall:output:0/^batch_normalization_74/StatefulPartitionedCall/^batch_normalization_75/StatefulPartitionedCall/^batch_normalization_76/StatefulPartitionedCall/^batch_normalization_77/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::2`
.batch_normalization_74/StatefulPartitionedCall.batch_normalization_74/StatefulPartitionedCall2`
.batch_normalization_75/StatefulPartitionedCall.batch_normalization_75/StatefulPartitionedCall2`
.batch_normalization_76/StatefulPartitionedCall.batch_normalization_76/StatefulPartitionedCall2`
.batch_normalization_77/StatefulPartitionedCall.batch_normalization_77/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ж
А
7__inference_batch_normalization_74_layer_call_fn_376470

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_3747802
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Р
С
4__inference_conv2d_transpose_29_layer_call_fn_375342

inputs"
statefulpartitionedcall_args_1
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3753352
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+                           @:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╩$
Э
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_376697

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_376682
assignmovingavg_1_376689
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
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
Const_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/376682*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xп
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/376682*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_376682*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/376682*
_output_shapes	
:А2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/376682*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_376682AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/376682*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/376689*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╖
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/376689*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_376689*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/376689*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/376689*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_376689AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/376689*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╣
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
В
ї
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_376815

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ч
Й
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_376604

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOp^
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

LogicalAndУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Аb2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аb2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Аb2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аb2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         Аb::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
┤/
╔
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_376438

inputs
assignmovingavg_376413
assignmovingavg_1_376419)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOp^
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

LogicalAndК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/376413*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_376413*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/376413*
_output_shapes	
:А2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/376413*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_376413AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/376413*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/376419*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_376419*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/376419*
_output_shapes	
:А2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/376419*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_376419AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/376419*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
ю
▌
D__inference_dense_33_layer_call_and_return_conditional_losses_375356

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ў
╛
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_374994

inputs,
(conv2d_transpose_readvariableop_resource
identityИвconv2d_transpose/ReadVariableOpD
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
strided_slice/stack_2т
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
strided_slice_1/stack_2ь
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
strided_slice_2/stack_2ь
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
B :А2	
stack/3В
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
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3╡
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
conv2d_transposeк
IdentityIdentityconv2d_transpose:output:0 ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,                           А:2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
Ї
А
7__inference_batch_normalization_76_layer_call_fn_376737

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_3751262
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
щ
f
J__inference_leaky_re_lu_74_layer_call_and_return_conditional_losses_375552

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           @2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:& "
 
_user_specified_nameinputs
▒
K
/__inference_leaky_re_lu_74_layer_call_fn_376843

inputs
identity╧
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_74_layer_call_and_return_conditional_losses_3755522
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:& "
 
_user_specified_nameinputs
╩$
Э
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_375095

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_375080
assignmovingavg_1_375087
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
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
Const_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/375080*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xп
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/375080*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_375080*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/375080*
_output_shapes	
:А2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/375080*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_375080AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/375080*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/375087*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╖
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/375087*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_375087*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/375087*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/375087*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_375087AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/375087*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╣
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
иЪ
м
I__inference_sequential_19_layer_call_and_return_conditional_losses_376107

inputs+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource1
-batch_normalization_74_assignmovingavg_3758883
/batch_normalization_74_assignmovingavg_1_375894@
<batch_normalization_74_batchnorm_mul_readvariableop_resource<
8batch_normalization_74_batchnorm_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource1
-batch_normalization_75_assignmovingavg_3759303
/batch_normalization_75_assignmovingavg_1_375936@
<batch_normalization_75_batchnorm_mul_readvariableop_resource<
8batch_normalization_75_batchnorm_readvariableop_resource@
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource2
.batch_normalization_76_readvariableop_resource4
0batch_normalization_76_readvariableop_1_resource1
-batch_normalization_76_assignmovingavg_3760073
/batch_normalization_76_assignmovingavg_1_376014@
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource2
.batch_normalization_77_readvariableop_resource4
0batch_normalization_77_readvariableop_1_resource1
-batch_normalization_77_assignmovingavg_3760643
/batch_normalization_77_assignmovingavg_1_376071@
<conv2d_transpose_29_conv2d_transpose_readvariableop_resource
identityИв:batch_normalization_74/AssignMovingAvg/AssignSubVariableOpв5batch_normalization_74/AssignMovingAvg/ReadVariableOpв<batch_normalization_74/AssignMovingAvg_1/AssignSubVariableOpв7batch_normalization_74/AssignMovingAvg_1/ReadVariableOpв/batch_normalization_74/batchnorm/ReadVariableOpв3batch_normalization_74/batchnorm/mul/ReadVariableOpв:batch_normalization_75/AssignMovingAvg/AssignSubVariableOpв5batch_normalization_75/AssignMovingAvg/ReadVariableOpв<batch_normalization_75/AssignMovingAvg_1/AssignSubVariableOpв7batch_normalization_75/AssignMovingAvg_1/ReadVariableOpв/batch_normalization_75/batchnorm/ReadVariableOpв3batch_normalization_75/batchnorm/mul/ReadVariableOpв:batch_normalization_76/AssignMovingAvg/AssignSubVariableOpв5batch_normalization_76/AssignMovingAvg/ReadVariableOpв<batch_normalization_76/AssignMovingAvg_1/AssignSubVariableOpв7batch_normalization_76/AssignMovingAvg_1/ReadVariableOpв%batch_normalization_76/ReadVariableOpв'batch_normalization_76/ReadVariableOp_1в:batch_normalization_77/AssignMovingAvg/AssignSubVariableOpв5batch_normalization_77/AssignMovingAvg/ReadVariableOpв<batch_normalization_77/AssignMovingAvg_1/AssignSubVariableOpв7batch_normalization_77/AssignMovingAvg_1/ReadVariableOpв%batch_normalization_77/ReadVariableOpв'batch_normalization_77/ReadVariableOp_1в3conv2d_transpose_27/conv2d_transpose/ReadVariableOpв3conv2d_transpose_28/conv2d_transpose/ReadVariableOpв3conv2d_transpose_29/conv2d_transpose/ReadVariableOpвdense_33/BiasAdd/ReadVariableOpвdense_33/MatMul/ReadVariableOpвdense_34/BiasAdd/ReadVariableOpвdense_34/MatMul/ReadVariableOpй
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	dА*
dtype02 
dense_33/MatMul/ReadVariableOpП
dense_33/MatMulMatMulinputs&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_33/MatMulи
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_33/BiasAdd/ReadVariableOpж
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_33/BiasAddМ
#batch_normalization_74/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_74/LogicalAnd/xМ
#batch_normalization_74/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_74/LogicalAnd/y╚
!batch_normalization_74/LogicalAnd
LogicalAnd,batch_normalization_74/LogicalAnd/x:output:0,batch_normalization_74/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_74/LogicalAnd╕
5batch_normalization_74/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_74/moments/mean/reduction_indicesш
#batch_normalization_74/moments/meanMeandense_33/BiasAdd:output:0>batch_normalization_74/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2%
#batch_normalization_74/moments/mean┬
+batch_normalization_74/moments/StopGradientStopGradient,batch_normalization_74/moments/mean:output:0*
T0*
_output_shapes
:	А2-
+batch_normalization_74/moments/StopGradient¤
0batch_normalization_74/moments/SquaredDifferenceSquaredDifferencedense_33/BiasAdd:output:04batch_normalization_74/moments/StopGradient:output:0*
T0*(
_output_shapes
:         А22
0batch_normalization_74/moments/SquaredDifference└
9batch_normalization_74/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_74/moments/variance/reduction_indicesП
'batch_normalization_74/moments/varianceMean4batch_normalization_74/moments/SquaredDifference:z:0Bbatch_normalization_74/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2)
'batch_normalization_74/moments/variance╞
&batch_normalization_74/moments/SqueezeSqueeze,batch_normalization_74/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2(
&batch_normalization_74/moments/Squeeze╬
(batch_normalization_74/moments/Squeeze_1Squeeze0batch_normalization_74/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2*
(batch_normalization_74/moments/Squeeze_1у
,batch_normalization_74/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_74/AssignMovingAvg/375888*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2.
,batch_normalization_74/AssignMovingAvg/decay┘
5batch_normalization_74/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_74_assignmovingavg_375888*
_output_shapes	
:А*
dtype027
5batch_normalization_74/AssignMovingAvg/ReadVariableOp╖
*batch_normalization_74/AssignMovingAvg/subSub=batch_normalization_74/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_74/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_74/AssignMovingAvg/375888*
_output_shapes	
:А2,
*batch_normalization_74/AssignMovingAvg/subо
*batch_normalization_74/AssignMovingAvg/mulMul.batch_normalization_74/AssignMovingAvg/sub:z:05batch_normalization_74/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_74/AssignMovingAvg/375888*
_output_shapes	
:А2,
*batch_normalization_74/AssignMovingAvg/mulЛ
:batch_normalization_74/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_74_assignmovingavg_375888.batch_normalization_74/AssignMovingAvg/mul:z:06^batch_normalization_74/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_74/AssignMovingAvg/375888*
_output_shapes
 *
dtype02<
:batch_normalization_74/AssignMovingAvg/AssignSubVariableOpщ
.batch_normalization_74/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_74/AssignMovingAvg_1/375894*
_output_shapes
: *
dtype0*
valueB
 *
╫#<20
.batch_normalization_74/AssignMovingAvg_1/decay▀
7batch_normalization_74/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_74_assignmovingavg_1_375894*
_output_shapes	
:А*
dtype029
7batch_normalization_74/AssignMovingAvg_1/ReadVariableOp┴
,batch_normalization_74/AssignMovingAvg_1/subSub?batch_normalization_74/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_74/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_74/AssignMovingAvg_1/375894*
_output_shapes	
:А2.
,batch_normalization_74/AssignMovingAvg_1/sub╕
,batch_normalization_74/AssignMovingAvg_1/mulMul0batch_normalization_74/AssignMovingAvg_1/sub:z:07batch_normalization_74/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_74/AssignMovingAvg_1/375894*
_output_shapes	
:А2.
,batch_normalization_74/AssignMovingAvg_1/mulЧ
<batch_normalization_74/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_74_assignmovingavg_1_3758940batch_normalization_74/AssignMovingAvg_1/mul:z:08^batch_normalization_74/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_74/AssignMovingAvg_1/375894*
_output_shapes
 *
dtype02>
<batch_normalization_74/AssignMovingAvg_1/AssignSubVariableOpХ
&batch_normalization_74/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2(
&batch_normalization_74/batchnorm/add/y▀
$batch_normalization_74/batchnorm/addAddV21batch_normalization_74/moments/Squeeze_1:output:0/batch_normalization_74/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2&
$batch_normalization_74/batchnorm/addй
&batch_normalization_74/batchnorm/RsqrtRsqrt(batch_normalization_74/batchnorm/add:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_74/batchnorm/Rsqrtф
3batch_normalization_74/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_74_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3batch_normalization_74/batchnorm/mul/ReadVariableOpт
$batch_normalization_74/batchnorm/mulMul*batch_normalization_74/batchnorm/Rsqrt:y:0;batch_normalization_74/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2&
$batch_normalization_74/batchnorm/mul╧
&batch_normalization_74/batchnorm/mul_1Muldense_33/BiasAdd:output:0(batch_normalization_74/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_74/batchnorm/mul_1╪
&batch_normalization_74/batchnorm/mul_2Mul/batch_normalization_74/moments/Squeeze:output:0(batch_normalization_74/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_74/batchnorm/mul_2╪
/batch_normalization_74/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_74_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/batch_normalization_74/batchnorm/ReadVariableOp▐
$batch_normalization_74/batchnorm/subSub7batch_normalization_74/batchnorm/ReadVariableOp:value:0*batch_normalization_74/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2&
$batch_normalization_74/batchnorm/subт
&batch_normalization_74/batchnorm/add_1AddV2*batch_normalization_74/batchnorm/mul_1:z:0(batch_normalization_74/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_74/batchnorm/add_1Ч
leaky_re_lu_71/LeakyRelu	LeakyRelu*batch_normalization_74/batchnorm/add_1:z:0*(
_output_shapes
:         А2
leaky_re_lu_71/LeakyReluк
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource* 
_output_shapes
:
ААb*
dtype02 
dense_34/MatMul/ReadVariableOpп
dense_34/MatMulMatMul&leaky_re_lu_71/LeakyRelu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аb2
dense_34/MatMulи
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes	
:Аb*
dtype02!
dense_34/BiasAdd/ReadVariableOpж
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аb2
dense_34/BiasAddМ
#batch_normalization_75/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_75/LogicalAnd/xМ
#batch_normalization_75/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_75/LogicalAnd/y╚
!batch_normalization_75/LogicalAnd
LogicalAnd,batch_normalization_75/LogicalAnd/x:output:0,batch_normalization_75/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_75/LogicalAnd╕
5batch_normalization_75/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_75/moments/mean/reduction_indicesш
#batch_normalization_75/moments/meanMeandense_34/BiasAdd:output:0>batch_normalization_75/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Аb*
	keep_dims(2%
#batch_normalization_75/moments/mean┬
+batch_normalization_75/moments/StopGradientStopGradient,batch_normalization_75/moments/mean:output:0*
T0*
_output_shapes
:	Аb2-
+batch_normalization_75/moments/StopGradient¤
0batch_normalization_75/moments/SquaredDifferenceSquaredDifferencedense_34/BiasAdd:output:04batch_normalization_75/moments/StopGradient:output:0*
T0*(
_output_shapes
:         Аb22
0batch_normalization_75/moments/SquaredDifference└
9batch_normalization_75/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_75/moments/variance/reduction_indicesП
'batch_normalization_75/moments/varianceMean4batch_normalization_75/moments/SquaredDifference:z:0Bbatch_normalization_75/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Аb*
	keep_dims(2)
'batch_normalization_75/moments/variance╞
&batch_normalization_75/moments/SqueezeSqueeze,batch_normalization_75/moments/mean:output:0*
T0*
_output_shapes	
:Аb*
squeeze_dims
 2(
&batch_normalization_75/moments/Squeeze╬
(batch_normalization_75/moments/Squeeze_1Squeeze0batch_normalization_75/moments/variance:output:0*
T0*
_output_shapes	
:Аb*
squeeze_dims
 2*
(batch_normalization_75/moments/Squeeze_1у
,batch_normalization_75/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_75/AssignMovingAvg/375930*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2.
,batch_normalization_75/AssignMovingAvg/decay┘
5batch_normalization_75/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_75_assignmovingavg_375930*
_output_shapes	
:Аb*
dtype027
5batch_normalization_75/AssignMovingAvg/ReadVariableOp╖
*batch_normalization_75/AssignMovingAvg/subSub=batch_normalization_75/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_75/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_75/AssignMovingAvg/375930*
_output_shapes	
:Аb2,
*batch_normalization_75/AssignMovingAvg/subо
*batch_normalization_75/AssignMovingAvg/mulMul.batch_normalization_75/AssignMovingAvg/sub:z:05batch_normalization_75/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_75/AssignMovingAvg/375930*
_output_shapes	
:Аb2,
*batch_normalization_75/AssignMovingAvg/mulЛ
:batch_normalization_75/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_75_assignmovingavg_375930.batch_normalization_75/AssignMovingAvg/mul:z:06^batch_normalization_75/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_75/AssignMovingAvg/375930*
_output_shapes
 *
dtype02<
:batch_normalization_75/AssignMovingAvg/AssignSubVariableOpщ
.batch_normalization_75/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_75/AssignMovingAvg_1/375936*
_output_shapes
: *
dtype0*
valueB
 *
╫#<20
.batch_normalization_75/AssignMovingAvg_1/decay▀
7batch_normalization_75/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_75_assignmovingavg_1_375936*
_output_shapes	
:Аb*
dtype029
7batch_normalization_75/AssignMovingAvg_1/ReadVariableOp┴
,batch_normalization_75/AssignMovingAvg_1/subSub?batch_normalization_75/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_75/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_75/AssignMovingAvg_1/375936*
_output_shapes	
:Аb2.
,batch_normalization_75/AssignMovingAvg_1/sub╕
,batch_normalization_75/AssignMovingAvg_1/mulMul0batch_normalization_75/AssignMovingAvg_1/sub:z:07batch_normalization_75/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_75/AssignMovingAvg_1/375936*
_output_shapes	
:Аb2.
,batch_normalization_75/AssignMovingAvg_1/mulЧ
<batch_normalization_75/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_75_assignmovingavg_1_3759360batch_normalization_75/AssignMovingAvg_1/mul:z:08^batch_normalization_75/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_75/AssignMovingAvg_1/375936*
_output_shapes
 *
dtype02>
<batch_normalization_75/AssignMovingAvg_1/AssignSubVariableOpХ
&batch_normalization_75/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2(
&batch_normalization_75/batchnorm/add/y▀
$batch_normalization_75/batchnorm/addAddV21batch_normalization_75/moments/Squeeze_1:output:0/batch_normalization_75/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Аb2&
$batch_normalization_75/batchnorm/addй
&batch_normalization_75/batchnorm/RsqrtRsqrt(batch_normalization_75/batchnorm/add:z:0*
T0*
_output_shapes	
:Аb2(
&batch_normalization_75/batchnorm/Rsqrtф
3batch_normalization_75/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_75_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Аb*
dtype025
3batch_normalization_75/batchnorm/mul/ReadVariableOpт
$batch_normalization_75/batchnorm/mulMul*batch_normalization_75/batchnorm/Rsqrt:y:0;batch_normalization_75/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аb2&
$batch_normalization_75/batchnorm/mul╧
&batch_normalization_75/batchnorm/mul_1Muldense_34/BiasAdd:output:0(batch_normalization_75/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Аb2(
&batch_normalization_75/batchnorm/mul_1╪
&batch_normalization_75/batchnorm/mul_2Mul/batch_normalization_75/moments/Squeeze:output:0(batch_normalization_75/batchnorm/mul:z:0*
T0*
_output_shapes	
:Аb2(
&batch_normalization_75/batchnorm/mul_2╪
/batch_normalization_75/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_75_batchnorm_readvariableop_resource*
_output_shapes	
:Аb*
dtype021
/batch_normalization_75/batchnorm/ReadVariableOp▐
$batch_normalization_75/batchnorm/subSub7batch_normalization_75/batchnorm/ReadVariableOp:value:0*batch_normalization_75/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аb2&
$batch_normalization_75/batchnorm/subт
&batch_normalization_75/batchnorm/add_1AddV2*batch_normalization_75/batchnorm/mul_1:z:0(batch_normalization_75/batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аb2(
&batch_normalization_75/batchnorm/add_1Ч
leaky_re_lu_72/LeakyRelu	LeakyRelu*batch_normalization_75/batchnorm/add_1:z:0*(
_output_shapes
:         Аb2
leaky_re_lu_72/LeakyRelux
reshape_9/ShapeShape&leaky_re_lu_72/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_9/ShapeИ
reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_9/strided_slice/stackМ
reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_1М
reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_2Ю
reshape_9/strided_sliceStridedSlicereshape_9/Shape:output:0&reshape_9/strided_slice/stack:output:0(reshape_9/strided_slice/stack_1:output:0(reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_9/strided_slicex
reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/1x
reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/2y
reshape_9/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :А2
reshape_9/Reshape/shape/3Ў
reshape_9/Reshape/shapePack reshape_9/strided_slice:output:0"reshape_9/Reshape/shape/1:output:0"reshape_9/Reshape/shape/2:output:0"reshape_9/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_9/Reshape/shape╢
reshape_9/ReshapeReshape&leaky_re_lu_72/LeakyRelu:activations:0 reshape_9/Reshape/shape:output:0*
T0*0
_output_shapes
:         А2
reshape_9/ReshapeА
conv2d_transpose_27/ShapeShapereshape_9/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_27/ShapeЬ
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_27/strided_slice/stackа
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice/stack_1а
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice/stack_2┌
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_27/strided_sliceа
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice_1/stackд
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_1/stack_1д
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_1/stack_2ф
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/Shape:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_27/strided_slice_1а
)conv2d_transpose_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice_2/stackд
+conv2d_transpose_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_2/stack_1д
+conv2d_transpose_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_2/stack_2ф
#conv2d_transpose_27/strided_slice_2StridedSlice"conv2d_transpose_27/Shape:output:02conv2d_transpose_27/strided_slice_2/stack:output:04conv2d_transpose_27/strided_slice_2/stack_1:output:04conv2d_transpose_27/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_27/strided_slice_2x
conv2d_transpose_27/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_27/mul/yм
conv2d_transpose_27/mulMul,conv2d_transpose_27/strided_slice_1:output:0"conv2d_transpose_27/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_27/mul|
conv2d_transpose_27/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_27/mul_1/y▓
conv2d_transpose_27/mul_1Mul,conv2d_transpose_27/strided_slice_2:output:0$conv2d_transpose_27/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_27/mul_1}
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2
conv2d_transpose_27/stack/3·
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0conv2d_transpose_27/mul:z:0conv2d_transpose_27/mul_1:z:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_27/stackа
)conv2d_transpose_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_27/strided_slice_3/stackд
+conv2d_transpose_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_3/stack_1д
+conv2d_transpose_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_3/stack_2ф
#conv2d_transpose_27/strided_slice_3StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_3/stack:output:04conv2d_transpose_27/strided_slice_3/stack_1:output:04conv2d_transpose_27/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_27/strided_slice_3ё
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype025
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp├
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0reshape_9/Reshape:output:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2&
$conv2d_transpose_27/conv2d_transposeМ
#batch_normalization_76/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_76/LogicalAnd/xМ
#batch_normalization_76/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_76/LogicalAnd/y╚
!batch_normalization_76/LogicalAnd
LogicalAnd,batch_normalization_76/LogicalAnd/x:output:0,batch_normalization_76/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_76/LogicalAnd║
%batch_normalization_76/ReadVariableOpReadVariableOp.batch_normalization_76_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_76/ReadVariableOp└
'batch_normalization_76/ReadVariableOp_1ReadVariableOp0batch_normalization_76_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_76/ReadVariableOp_1
batch_normalization_76/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_76/ConstГ
batch_normalization_76/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_76/Const_1╗
'batch_normalization_76/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_27/conv2d_transpose:output:0-batch_normalization_76/ReadVariableOp:value:0/batch_normalization_76/ReadVariableOp_1:value:0%batch_normalization_76/Const:output:0'batch_normalization_76/Const_1:output:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:2)
'batch_normalization_76/FusedBatchNormV3Е
batch_normalization_76/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2 
batch_normalization_76/Const_2у
,batch_normalization_76/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_76/AssignMovingAvg/376007*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,batch_normalization_76/AssignMovingAvg/sub/xв
*batch_normalization_76/AssignMovingAvg/subSub5batch_normalization_76/AssignMovingAvg/sub/x:output:0'batch_normalization_76/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_76/AssignMovingAvg/376007*
_output_shapes
: 2,
*batch_normalization_76/AssignMovingAvg/sub┘
5batch_normalization_76/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_76_assignmovingavg_376007*
_output_shapes	
:А*
dtype027
5batch_normalization_76/AssignMovingAvg/ReadVariableOp└
,batch_normalization_76/AssignMovingAvg/sub_1Sub=batch_normalization_76/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_76/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_76/AssignMovingAvg/376007*
_output_shapes	
:А2.
,batch_normalization_76/AssignMovingAvg/sub_1й
*batch_normalization_76/AssignMovingAvg/mulMul0batch_normalization_76/AssignMovingAvg/sub_1:z:0.batch_normalization_76/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_76/AssignMovingAvg/376007*
_output_shapes	
:А2,
*batch_normalization_76/AssignMovingAvg/mulЛ
:batch_normalization_76/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_76_assignmovingavg_376007.batch_normalization_76/AssignMovingAvg/mul:z:06^batch_normalization_76/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_76/AssignMovingAvg/376007*
_output_shapes
 *
dtype02<
:batch_normalization_76/AssignMovingAvg/AssignSubVariableOpщ
.batch_normalization_76/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_76/AssignMovingAvg_1/376014*
_output_shapes
: *
dtype0*
valueB
 *  А?20
.batch_normalization_76/AssignMovingAvg_1/sub/xк
,batch_normalization_76/AssignMovingAvg_1/subSub7batch_normalization_76/AssignMovingAvg_1/sub/x:output:0'batch_normalization_76/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_76/AssignMovingAvg_1/376014*
_output_shapes
: 2.
,batch_normalization_76/AssignMovingAvg_1/sub▀
7batch_normalization_76/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_76_assignmovingavg_1_376014*
_output_shapes	
:А*
dtype029
7batch_normalization_76/AssignMovingAvg_1/ReadVariableOp╠
.batch_normalization_76/AssignMovingAvg_1/sub_1Sub?batch_normalization_76/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_76/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_76/AssignMovingAvg_1/376014*
_output_shapes	
:А20
.batch_normalization_76/AssignMovingAvg_1/sub_1│
,batch_normalization_76/AssignMovingAvg_1/mulMul2batch_normalization_76/AssignMovingAvg_1/sub_1:z:00batch_normalization_76/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_76/AssignMovingAvg_1/376014*
_output_shapes	
:А2.
,batch_normalization_76/AssignMovingAvg_1/mulЧ
<batch_normalization_76/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_76_assignmovingavg_1_3760140batch_normalization_76/AssignMovingAvg_1/mul:z:08^batch_normalization_76/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_76/AssignMovingAvg_1/376014*
_output_shapes
 *
dtype02>
<batch_normalization_76/AssignMovingAvg_1/AssignSubVariableOpа
leaky_re_lu_73/LeakyRelu	LeakyRelu+batch_normalization_76/FusedBatchNormV3:y:0*0
_output_shapes
:         А2
leaky_re_lu_73/LeakyReluМ
conv2d_transpose_28/ShapeShape&leaky_re_lu_73/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_28/ShapeЬ
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_28/strided_slice/stackа
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice/stack_1а
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice/stack_2┌
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_28/strided_sliceа
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice_1/stackд
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_1/stack_1д
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_1/stack_2ф
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/Shape:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_28/strided_slice_1а
)conv2d_transpose_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice_2/stackд
+conv2d_transpose_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_2/stack_1д
+conv2d_transpose_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_2/stack_2ф
#conv2d_transpose_28/strided_slice_2StridedSlice"conv2d_transpose_28/Shape:output:02conv2d_transpose_28/strided_slice_2/stack:output:04conv2d_transpose_28/strided_slice_2/stack_1:output:04conv2d_transpose_28/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_28/strided_slice_2x
conv2d_transpose_28/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_28/mul/yм
conv2d_transpose_28/mulMul,conv2d_transpose_28/strided_slice_1:output:0"conv2d_transpose_28/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_28/mul|
conv2d_transpose_28/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_28/mul_1/y▓
conv2d_transpose_28/mul_1Mul,conv2d_transpose_28/strided_slice_2:output:0$conv2d_transpose_28/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_28/mul_1|
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_28/stack/3·
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0conv2d_transpose_28/mul:z:0conv2d_transpose_28/mul_1:z:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_28/stackа
)conv2d_transpose_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_28/strided_slice_3/stackд
+conv2d_transpose_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_3/stack_1д
+conv2d_transpose_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_3/stack_2ф
#conv2d_transpose_28/strided_slice_3StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_3/stack:output:04conv2d_transpose_28/strided_slice_3/stack_1:output:04conv2d_transpose_28/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_28/strided_slice_3Ё
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype025
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp╬
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_73/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2&
$conv2d_transpose_28/conv2d_transposeМ
#batch_normalization_77/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_77/LogicalAnd/xМ
#batch_normalization_77/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_77/LogicalAnd/y╚
!batch_normalization_77/LogicalAnd
LogicalAnd,batch_normalization_77/LogicalAnd/x:output:0,batch_normalization_77/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_77/LogicalAnd╣
%batch_normalization_77/ReadVariableOpReadVariableOp.batch_normalization_77_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_77/ReadVariableOp┐
'batch_normalization_77/ReadVariableOp_1ReadVariableOp0batch_normalization_77_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_77/ReadVariableOp_1
batch_normalization_77/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_77/ConstГ
batch_normalization_77/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_77/Const_1╢
'batch_normalization_77/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_28/conv2d_transpose:output:0-batch_normalization_77/ReadVariableOp:value:0/batch_normalization_77/ReadVariableOp_1:value:0%batch_normalization_77/Const:output:0'batch_normalization_77/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:2)
'batch_normalization_77/FusedBatchNormV3Е
batch_normalization_77/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2 
batch_normalization_77/Const_2у
,batch_normalization_77/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_77/AssignMovingAvg/376064*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,batch_normalization_77/AssignMovingAvg/sub/xв
*batch_normalization_77/AssignMovingAvg/subSub5batch_normalization_77/AssignMovingAvg/sub/x:output:0'batch_normalization_77/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_77/AssignMovingAvg/376064*
_output_shapes
: 2,
*batch_normalization_77/AssignMovingAvg/sub╪
5batch_normalization_77/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_77_assignmovingavg_376064*
_output_shapes
:@*
dtype027
5batch_normalization_77/AssignMovingAvg/ReadVariableOp┐
,batch_normalization_77/AssignMovingAvg/sub_1Sub=batch_normalization_77/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_77/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_77/AssignMovingAvg/376064*
_output_shapes
:@2.
,batch_normalization_77/AssignMovingAvg/sub_1и
*batch_normalization_77/AssignMovingAvg/mulMul0batch_normalization_77/AssignMovingAvg/sub_1:z:0.batch_normalization_77/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_77/AssignMovingAvg/376064*
_output_shapes
:@2,
*batch_normalization_77/AssignMovingAvg/mulЛ
:batch_normalization_77/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_77_assignmovingavg_376064.batch_normalization_77/AssignMovingAvg/mul:z:06^batch_normalization_77/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_77/AssignMovingAvg/376064*
_output_shapes
 *
dtype02<
:batch_normalization_77/AssignMovingAvg/AssignSubVariableOpщ
.batch_normalization_77/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_77/AssignMovingAvg_1/376071*
_output_shapes
: *
dtype0*
valueB
 *  А?20
.batch_normalization_77/AssignMovingAvg_1/sub/xк
,batch_normalization_77/AssignMovingAvg_1/subSub7batch_normalization_77/AssignMovingAvg_1/sub/x:output:0'batch_normalization_77/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_77/AssignMovingAvg_1/376071*
_output_shapes
: 2.
,batch_normalization_77/AssignMovingAvg_1/sub▐
7batch_normalization_77/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_77_assignmovingavg_1_376071*
_output_shapes
:@*
dtype029
7batch_normalization_77/AssignMovingAvg_1/ReadVariableOp╦
.batch_normalization_77/AssignMovingAvg_1/sub_1Sub?batch_normalization_77/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_77/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_77/AssignMovingAvg_1/376071*
_output_shapes
:@20
.batch_normalization_77/AssignMovingAvg_1/sub_1▓
,batch_normalization_77/AssignMovingAvg_1/mulMul2batch_normalization_77/AssignMovingAvg_1/sub_1:z:00batch_normalization_77/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_77/AssignMovingAvg_1/376071*
_output_shapes
:@2.
,batch_normalization_77/AssignMovingAvg_1/mulЧ
<batch_normalization_77/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_77_assignmovingavg_1_3760710batch_normalization_77/AssignMovingAvg_1/mul:z:08^batch_normalization_77/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_77/AssignMovingAvg_1/376071*
_output_shapes
 *
dtype02>
<batch_normalization_77/AssignMovingAvg_1/AssignSubVariableOpЯ
leaky_re_lu_74/LeakyRelu	LeakyRelu+batch_normalization_77/FusedBatchNormV3:y:0*/
_output_shapes
:         @2
leaky_re_lu_74/LeakyReluМ
conv2d_transpose_29/ShapeShape&leaky_re_lu_74/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_29/ShapeЬ
'conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_29/strided_slice/stackа
)conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_29/strided_slice/stack_1а
)conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_29/strided_slice/stack_2┌
!conv2d_transpose_29/strided_sliceStridedSlice"conv2d_transpose_29/Shape:output:00conv2d_transpose_29/strided_slice/stack:output:02conv2d_transpose_29/strided_slice/stack_1:output:02conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_29/strided_sliceа
)conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_29/strided_slice_1/stackд
+conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_1/stack_1д
+conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_1/stack_2ф
#conv2d_transpose_29/strided_slice_1StridedSlice"conv2d_transpose_29/Shape:output:02conv2d_transpose_29/strided_slice_1/stack:output:04conv2d_transpose_29/strided_slice_1/stack_1:output:04conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_29/strided_slice_1а
)conv2d_transpose_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_29/strided_slice_2/stackд
+conv2d_transpose_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_2/stack_1д
+conv2d_transpose_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_2/stack_2ф
#conv2d_transpose_29/strided_slice_2StridedSlice"conv2d_transpose_29/Shape:output:02conv2d_transpose_29/strided_slice_2/stack:output:04conv2d_transpose_29/strided_slice_2/stack_1:output:04conv2d_transpose_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_29/strided_slice_2x
conv2d_transpose_29/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_29/mul/yм
conv2d_transpose_29/mulMul,conv2d_transpose_29/strided_slice_1:output:0"conv2d_transpose_29/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_29/mul|
conv2d_transpose_29/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_29/mul_1/y▓
conv2d_transpose_29/mul_1Mul,conv2d_transpose_29/strided_slice_2:output:0$conv2d_transpose_29/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_29/mul_1|
conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_29/stack/3·
conv2d_transpose_29/stackPack*conv2d_transpose_29/strided_slice:output:0conv2d_transpose_29/mul:z:0conv2d_transpose_29/mul_1:z:0$conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_29/stackа
)conv2d_transpose_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_29/strided_slice_3/stackд
+conv2d_transpose_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_3/stack_1д
+conv2d_transpose_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_29/strided_slice_3/stack_2ф
#conv2d_transpose_29/strided_slice_3StridedSlice"conv2d_transpose_29/stack:output:02conv2d_transpose_29/strided_slice_3/stack:output:04conv2d_transpose_29/strided_slice_3/stack_1:output:04conv2d_transpose_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_29/strided_slice_3я
3conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype025
3conv2d_transpose_29/conv2d_transpose/ReadVariableOp╬
$conv2d_transpose_29/conv2d_transposeConv2DBackpropInput"conv2d_transpose_29/stack:output:0;conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_74/LeakyRelu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2&
$conv2d_transpose_29/conv2d_transposeе
conv2d_transpose_29/TanhTanh-conv2d_transpose_29/conv2d_transpose:output:0*
T0*/
_output_shapes
:         2
conv2d_transpose_29/Tanh╠
IdentityIdentityconv2d_transpose_29/Tanh:y:0;^batch_normalization_74/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_74/AssignMovingAvg/ReadVariableOp=^batch_normalization_74/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_74/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_74/batchnorm/ReadVariableOp4^batch_normalization_74/batchnorm/mul/ReadVariableOp;^batch_normalization_75/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_75/AssignMovingAvg/ReadVariableOp=^batch_normalization_75/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_75/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_75/batchnorm/ReadVariableOp4^batch_normalization_75/batchnorm/mul/ReadVariableOp;^batch_normalization_76/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_76/AssignMovingAvg/ReadVariableOp=^batch_normalization_76/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_76/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_76/ReadVariableOp(^batch_normalization_76/ReadVariableOp_1;^batch_normalization_77/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_77/AssignMovingAvg/ReadVariableOp=^batch_normalization_77/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_77/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_77/ReadVariableOp(^batch_normalization_77/ReadVariableOp_14^conv2d_transpose_27/conv2d_transpose/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp4^conv2d_transpose_29/conv2d_transpose/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::2x
:batch_normalization_74/AssignMovingAvg/AssignSubVariableOp:batch_normalization_74/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_74/AssignMovingAvg/ReadVariableOp5batch_normalization_74/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_74/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_74/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_74/AssignMovingAvg_1/ReadVariableOp7batch_normalization_74/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_74/batchnorm/ReadVariableOp/batch_normalization_74/batchnorm/ReadVariableOp2j
3batch_normalization_74/batchnorm/mul/ReadVariableOp3batch_normalization_74/batchnorm/mul/ReadVariableOp2x
:batch_normalization_75/AssignMovingAvg/AssignSubVariableOp:batch_normalization_75/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_75/AssignMovingAvg/ReadVariableOp5batch_normalization_75/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_75/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_75/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_75/AssignMovingAvg_1/ReadVariableOp7batch_normalization_75/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_75/batchnorm/ReadVariableOp/batch_normalization_75/batchnorm/ReadVariableOp2j
3batch_normalization_75/batchnorm/mul/ReadVariableOp3batch_normalization_75/batchnorm/mul/ReadVariableOp2x
:batch_normalization_76/AssignMovingAvg/AssignSubVariableOp:batch_normalization_76/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_76/AssignMovingAvg/ReadVariableOp5batch_normalization_76/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_76/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_76/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_76/AssignMovingAvg_1/ReadVariableOp7batch_normalization_76/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_76/ReadVariableOp%batch_normalization_76/ReadVariableOp2R
'batch_normalization_76/ReadVariableOp_1'batch_normalization_76/ReadVariableOp_12x
:batch_normalization_77/AssignMovingAvg/AssignSubVariableOp:batch_normalization_77/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_77/AssignMovingAvg/ReadVariableOp5batch_normalization_77/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_77/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_77/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_77/AssignMovingAvg_1/ReadVariableOp7batch_normalization_77/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_77/ReadVariableOp%batch_normalization_77/ReadVariableOp2R
'batch_normalization_77/ReadVariableOp_1'batch_normalization_77/ReadVariableOp_12j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_29/conv2d_transpose/ReadVariableOp3conv2d_transpose_29/conv2d_transpose/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
х
╣
.__inference_sequential_19_layer_call_fn_375674
dense_33_input"
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
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCalldense_33_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_3756482
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namedense_33_input
ж
А
7__inference_batch_normalization_75_layer_call_fn_376613

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_3749242
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         Аb::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ё
А
7__inference_batch_normalization_77_layer_call_fn_376824

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_3752652
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
х
K
/__inference_leaky_re_lu_71_layer_call_fn_376489

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_3753962
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
ь
f
J__inference_leaky_re_lu_73_layer_call_and_return_conditional_losses_376742

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,                           А2
	LeakyReluЖ
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:& "
 
_user_specified_nameinputs
Ў╘
И
!__inference__wrapped_model_374675
dense_33_input9
5sequential_19_dense_33_matmul_readvariableop_resource:
6sequential_19_dense_33_biasadd_readvariableop_resourceJ
Fsequential_19_batch_normalization_74_batchnorm_readvariableop_resourceN
Jsequential_19_batch_normalization_74_batchnorm_mul_readvariableop_resourceL
Hsequential_19_batch_normalization_74_batchnorm_readvariableop_1_resourceL
Hsequential_19_batch_normalization_74_batchnorm_readvariableop_2_resource9
5sequential_19_dense_34_matmul_readvariableop_resource:
6sequential_19_dense_34_biasadd_readvariableop_resourceJ
Fsequential_19_batch_normalization_75_batchnorm_readvariableop_resourceN
Jsequential_19_batch_normalization_75_batchnorm_mul_readvariableop_resourceL
Hsequential_19_batch_normalization_75_batchnorm_readvariableop_1_resourceL
Hsequential_19_batch_normalization_75_batchnorm_readvariableop_2_resourceN
Jsequential_19_conv2d_transpose_27_conv2d_transpose_readvariableop_resource@
<sequential_19_batch_normalization_76_readvariableop_resourceB
>sequential_19_batch_normalization_76_readvariableop_1_resourceQ
Msequential_19_batch_normalization_76_fusedbatchnormv3_readvariableop_resourceS
Osequential_19_batch_normalization_76_fusedbatchnormv3_readvariableop_1_resourceN
Jsequential_19_conv2d_transpose_28_conv2d_transpose_readvariableop_resource@
<sequential_19_batch_normalization_77_readvariableop_resourceB
>sequential_19_batch_normalization_77_readvariableop_1_resourceQ
Msequential_19_batch_normalization_77_fusedbatchnormv3_readvariableop_resourceS
Osequential_19_batch_normalization_77_fusedbatchnormv3_readvariableop_1_resourceN
Jsequential_19_conv2d_transpose_29_conv2d_transpose_readvariableop_resource
identityИв=sequential_19/batch_normalization_74/batchnorm/ReadVariableOpв?sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_1в?sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_2вAsequential_19/batch_normalization_74/batchnorm/mul/ReadVariableOpв=sequential_19/batch_normalization_75/batchnorm/ReadVariableOpв?sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_1в?sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_2вAsequential_19/batch_normalization_75/batchnorm/mul/ReadVariableOpвDsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOpвFsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1в3sequential_19/batch_normalization_76/ReadVariableOpв5sequential_19/batch_normalization_76/ReadVariableOp_1вDsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOpвFsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1в3sequential_19/batch_normalization_77/ReadVariableOpв5sequential_19/batch_normalization_77/ReadVariableOp_1вAsequential_19/conv2d_transpose_27/conv2d_transpose/ReadVariableOpвAsequential_19/conv2d_transpose_28/conv2d_transpose/ReadVariableOpвAsequential_19/conv2d_transpose_29/conv2d_transpose/ReadVariableOpв-sequential_19/dense_33/BiasAdd/ReadVariableOpв,sequential_19/dense_33/MatMul/ReadVariableOpв-sequential_19/dense_34/BiasAdd/ReadVariableOpв,sequential_19/dense_34/MatMul/ReadVariableOp╙
,sequential_19/dense_33/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_33_matmul_readvariableop_resource*
_output_shapes
:	dА*
dtype02.
,sequential_19/dense_33/MatMul/ReadVariableOp┴
sequential_19/dense_33/MatMulMatMuldense_33_input4sequential_19/dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_19/dense_33/MatMul╥
-sequential_19/dense_33/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_33_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_19/dense_33/BiasAdd/ReadVariableOp▐
sequential_19/dense_33/BiasAddBiasAdd'sequential_19/dense_33/MatMul:product:05sequential_19/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
sequential_19/dense_33/BiasAddи
1sequential_19/batch_normalization_74/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1sequential_19/batch_normalization_74/LogicalAnd/xи
1sequential_19/batch_normalization_74/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1sequential_19/batch_normalization_74/LogicalAnd/yА
/sequential_19/batch_normalization_74/LogicalAnd
LogicalAnd:sequential_19/batch_normalization_74/LogicalAnd/x:output:0:sequential_19/batch_normalization_74/LogicalAnd/y:output:0*
_output_shapes
: 21
/sequential_19/batch_normalization_74/LogicalAndВ
=sequential_19/batch_normalization_74/batchnorm/ReadVariableOpReadVariableOpFsequential_19_batch_normalization_74_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02?
=sequential_19/batch_normalization_74/batchnorm/ReadVariableOp▒
4sequential_19/batch_normalization_74/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:26
4sequential_19/batch_normalization_74/batchnorm/add/yЭ
2sequential_19/batch_normalization_74/batchnorm/addAddV2Esequential_19/batch_normalization_74/batchnorm/ReadVariableOp:value:0=sequential_19/batch_normalization_74/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А24
2sequential_19/batch_normalization_74/batchnorm/add╙
4sequential_19/batch_normalization_74/batchnorm/RsqrtRsqrt6sequential_19/batch_normalization_74/batchnorm/add:z:0*
T0*
_output_shapes	
:А26
4sequential_19/batch_normalization_74/batchnorm/RsqrtО
Asequential_19/batch_normalization_74/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_19_batch_normalization_74_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02C
Asequential_19/batch_normalization_74/batchnorm/mul/ReadVariableOpЪ
2sequential_19/batch_normalization_74/batchnorm/mulMul8sequential_19/batch_normalization_74/batchnorm/Rsqrt:y:0Isequential_19/batch_normalization_74/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А24
2sequential_19/batch_normalization_74/batchnorm/mulЗ
4sequential_19/batch_normalization_74/batchnorm/mul_1Mul'sequential_19/dense_33/BiasAdd:output:06sequential_19/batch_normalization_74/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А26
4sequential_19/batch_normalization_74/batchnorm/mul_1И
?sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_19_batch_normalization_74_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02A
?sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_1Ъ
4sequential_19/batch_normalization_74/batchnorm/mul_2MulGsequential_19/batch_normalization_74/batchnorm/ReadVariableOp_1:value:06sequential_19/batch_normalization_74/batchnorm/mul:z:0*
T0*
_output_shapes	
:А26
4sequential_19/batch_normalization_74/batchnorm/mul_2И
?sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_19_batch_normalization_74_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02A
?sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_2Ш
2sequential_19/batch_normalization_74/batchnorm/subSubGsequential_19/batch_normalization_74/batchnorm/ReadVariableOp_2:value:08sequential_19/batch_normalization_74/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А24
2sequential_19/batch_normalization_74/batchnorm/subЪ
4sequential_19/batch_normalization_74/batchnorm/add_1AddV28sequential_19/batch_normalization_74/batchnorm/mul_1:z:06sequential_19/batch_normalization_74/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А26
4sequential_19/batch_normalization_74/batchnorm/add_1┴
&sequential_19/leaky_re_lu_71/LeakyRelu	LeakyRelu8sequential_19/batch_normalization_74/batchnorm/add_1:z:0*(
_output_shapes
:         А2(
&sequential_19/leaky_re_lu_71/LeakyRelu╘
,sequential_19/dense_34/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_34_matmul_readvariableop_resource* 
_output_shapes
:
ААb*
dtype02.
,sequential_19/dense_34/MatMul/ReadVariableOpч
sequential_19/dense_34/MatMulMatMul4sequential_19/leaky_re_lu_71/LeakyRelu:activations:04sequential_19/dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аb2
sequential_19/dense_34/MatMul╥
-sequential_19/dense_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_34_biasadd_readvariableop_resource*
_output_shapes	
:Аb*
dtype02/
-sequential_19/dense_34/BiasAdd/ReadVariableOp▐
sequential_19/dense_34/BiasAddBiasAdd'sequential_19/dense_34/MatMul:product:05sequential_19/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аb2 
sequential_19/dense_34/BiasAddи
1sequential_19/batch_normalization_75/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1sequential_19/batch_normalization_75/LogicalAnd/xи
1sequential_19/batch_normalization_75/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1sequential_19/batch_normalization_75/LogicalAnd/yА
/sequential_19/batch_normalization_75/LogicalAnd
LogicalAnd:sequential_19/batch_normalization_75/LogicalAnd/x:output:0:sequential_19/batch_normalization_75/LogicalAnd/y:output:0*
_output_shapes
: 21
/sequential_19/batch_normalization_75/LogicalAndВ
=sequential_19/batch_normalization_75/batchnorm/ReadVariableOpReadVariableOpFsequential_19_batch_normalization_75_batchnorm_readvariableop_resource*
_output_shapes	
:Аb*
dtype02?
=sequential_19/batch_normalization_75/batchnorm/ReadVariableOp▒
4sequential_19/batch_normalization_75/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:26
4sequential_19/batch_normalization_75/batchnorm/add/yЭ
2sequential_19/batch_normalization_75/batchnorm/addAddV2Esequential_19/batch_normalization_75/batchnorm/ReadVariableOp:value:0=sequential_19/batch_normalization_75/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Аb24
2sequential_19/batch_normalization_75/batchnorm/add╙
4sequential_19/batch_normalization_75/batchnorm/RsqrtRsqrt6sequential_19/batch_normalization_75/batchnorm/add:z:0*
T0*
_output_shapes	
:Аb26
4sequential_19/batch_normalization_75/batchnorm/RsqrtО
Asequential_19/batch_normalization_75/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_19_batch_normalization_75_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Аb*
dtype02C
Asequential_19/batch_normalization_75/batchnorm/mul/ReadVariableOpЪ
2sequential_19/batch_normalization_75/batchnorm/mulMul8sequential_19/batch_normalization_75/batchnorm/Rsqrt:y:0Isequential_19/batch_normalization_75/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аb24
2sequential_19/batch_normalization_75/batchnorm/mulЗ
4sequential_19/batch_normalization_75/batchnorm/mul_1Mul'sequential_19/dense_34/BiasAdd:output:06sequential_19/batch_normalization_75/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Аb26
4sequential_19/batch_normalization_75/batchnorm/mul_1И
?sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_19_batch_normalization_75_batchnorm_readvariableop_1_resource*
_output_shapes	
:Аb*
dtype02A
?sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_1Ъ
4sequential_19/batch_normalization_75/batchnorm/mul_2MulGsequential_19/batch_normalization_75/batchnorm/ReadVariableOp_1:value:06sequential_19/batch_normalization_75/batchnorm/mul:z:0*
T0*
_output_shapes	
:Аb26
4sequential_19/batch_normalization_75/batchnorm/mul_2И
?sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_19_batch_normalization_75_batchnorm_readvariableop_2_resource*
_output_shapes	
:Аb*
dtype02A
?sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_2Ш
2sequential_19/batch_normalization_75/batchnorm/subSubGsequential_19/batch_normalization_75/batchnorm/ReadVariableOp_2:value:08sequential_19/batch_normalization_75/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аb24
2sequential_19/batch_normalization_75/batchnorm/subЪ
4sequential_19/batch_normalization_75/batchnorm/add_1AddV28sequential_19/batch_normalization_75/batchnorm/mul_1:z:06sequential_19/batch_normalization_75/batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аb26
4sequential_19/batch_normalization_75/batchnorm/add_1┴
&sequential_19/leaky_re_lu_72/LeakyRelu	LeakyRelu8sequential_19/batch_normalization_75/batchnorm/add_1:z:0*(
_output_shapes
:         Аb2(
&sequential_19/leaky_re_lu_72/LeakyReluв
sequential_19/reshape_9/ShapeShape4sequential_19/leaky_re_lu_72/LeakyRelu:activations:0*
T0*
_output_shapes
:2
sequential_19/reshape_9/Shapeд
+sequential_19/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_19/reshape_9/strided_slice/stackи
-sequential_19/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_19/reshape_9/strided_slice/stack_1и
-sequential_19/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_19/reshape_9/strided_slice/stack_2Є
%sequential_19/reshape_9/strided_sliceStridedSlice&sequential_19/reshape_9/Shape:output:04sequential_19/reshape_9/strided_slice/stack:output:06sequential_19/reshape_9/strided_slice/stack_1:output:06sequential_19/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_19/reshape_9/strided_sliceФ
'sequential_19/reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_19/reshape_9/Reshape/shape/1Ф
'sequential_19/reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_19/reshape_9/Reshape/shape/2Х
'sequential_19/reshape_9/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :А2)
'sequential_19/reshape_9/Reshape/shape/3╩
%sequential_19/reshape_9/Reshape/shapePack.sequential_19/reshape_9/strided_slice:output:00sequential_19/reshape_9/Reshape/shape/1:output:00sequential_19/reshape_9/Reshape/shape/2:output:00sequential_19/reshape_9/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_19/reshape_9/Reshape/shapeю
sequential_19/reshape_9/ReshapeReshape4sequential_19/leaky_re_lu_72/LeakyRelu:activations:0.sequential_19/reshape_9/Reshape/shape:output:0*
T0*0
_output_shapes
:         А2!
sequential_19/reshape_9/Reshapeк
'sequential_19/conv2d_transpose_27/ShapeShape(sequential_19/reshape_9/Reshape:output:0*
T0*
_output_shapes
:2)
'sequential_19/conv2d_transpose_27/Shape╕
5sequential_19/conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_19/conv2d_transpose_27/strided_slice/stack╝
7sequential_19/conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_27/strided_slice/stack_1╝
7sequential_19/conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_27/strided_slice/stack_2о
/sequential_19/conv2d_transpose_27/strided_sliceStridedSlice0sequential_19/conv2d_transpose_27/Shape:output:0>sequential_19/conv2d_transpose_27/strided_slice/stack:output:0@sequential_19/conv2d_transpose_27/strided_slice/stack_1:output:0@sequential_19/conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_19/conv2d_transpose_27/strided_slice╝
7sequential_19/conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_27/strided_slice_1/stack└
9sequential_19/conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_27/strided_slice_1/stack_1└
9sequential_19/conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_27/strided_slice_1/stack_2╕
1sequential_19/conv2d_transpose_27/strided_slice_1StridedSlice0sequential_19/conv2d_transpose_27/Shape:output:0@sequential_19/conv2d_transpose_27/strided_slice_1/stack:output:0Bsequential_19/conv2d_transpose_27/strided_slice_1/stack_1:output:0Bsequential_19/conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_19/conv2d_transpose_27/strided_slice_1╝
7sequential_19/conv2d_transpose_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_27/strided_slice_2/stack└
9sequential_19/conv2d_transpose_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_27/strided_slice_2/stack_1└
9sequential_19/conv2d_transpose_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_27/strided_slice_2/stack_2╕
1sequential_19/conv2d_transpose_27/strided_slice_2StridedSlice0sequential_19/conv2d_transpose_27/Shape:output:0@sequential_19/conv2d_transpose_27/strided_slice_2/stack:output:0Bsequential_19/conv2d_transpose_27/strided_slice_2/stack_1:output:0Bsequential_19/conv2d_transpose_27/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_19/conv2d_transpose_27/strided_slice_2Ф
'sequential_19/conv2d_transpose_27/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_19/conv2d_transpose_27/mul/yф
%sequential_19/conv2d_transpose_27/mulMul:sequential_19/conv2d_transpose_27/strided_slice_1:output:00sequential_19/conv2d_transpose_27/mul/y:output:0*
T0*
_output_shapes
: 2'
%sequential_19/conv2d_transpose_27/mulШ
)sequential_19/conv2d_transpose_27/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_19/conv2d_transpose_27/mul_1/yъ
'sequential_19/conv2d_transpose_27/mul_1Mul:sequential_19/conv2d_transpose_27/strided_slice_2:output:02sequential_19/conv2d_transpose_27/mul_1/y:output:0*
T0*
_output_shapes
: 2)
'sequential_19/conv2d_transpose_27/mul_1Щ
)sequential_19/conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2+
)sequential_19/conv2d_transpose_27/stack/3╬
'sequential_19/conv2d_transpose_27/stackPack8sequential_19/conv2d_transpose_27/strided_slice:output:0)sequential_19/conv2d_transpose_27/mul:z:0+sequential_19/conv2d_transpose_27/mul_1:z:02sequential_19/conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_19/conv2d_transpose_27/stack╝
7sequential_19/conv2d_transpose_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_19/conv2d_transpose_27/strided_slice_3/stack└
9sequential_19/conv2d_transpose_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_27/strided_slice_3/stack_1└
9sequential_19/conv2d_transpose_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_27/strided_slice_3/stack_2╕
1sequential_19/conv2d_transpose_27/strided_slice_3StridedSlice0sequential_19/conv2d_transpose_27/stack:output:0@sequential_19/conv2d_transpose_27/strided_slice_3/stack:output:0Bsequential_19/conv2d_transpose_27/strided_slice_3/stack_1:output:0Bsequential_19/conv2d_transpose_27/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_19/conv2d_transpose_27/strided_slice_3Ы
Asequential_19/conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_19_conv2d_transpose_27_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype02C
Asequential_19/conv2d_transpose_27/conv2d_transpose/ReadVariableOpЙ
2sequential_19/conv2d_transpose_27/conv2d_transposeConv2DBackpropInput0sequential_19/conv2d_transpose_27/stack:output:0Isequential_19/conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0(sequential_19/reshape_9/Reshape:output:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
24
2sequential_19/conv2d_transpose_27/conv2d_transposeи
1sequential_19/batch_normalization_76/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1sequential_19/batch_normalization_76/LogicalAnd/xи
1sequential_19/batch_normalization_76/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1sequential_19/batch_normalization_76/LogicalAnd/yА
/sequential_19/batch_normalization_76/LogicalAnd
LogicalAnd:sequential_19/batch_normalization_76/LogicalAnd/x:output:0:sequential_19/batch_normalization_76/LogicalAnd/y:output:0*
_output_shapes
: 21
/sequential_19/batch_normalization_76/LogicalAndф
3sequential_19/batch_normalization_76/ReadVariableOpReadVariableOp<sequential_19_batch_normalization_76_readvariableop_resource*
_output_shapes	
:А*
dtype025
3sequential_19/batch_normalization_76/ReadVariableOpъ
5sequential_19/batch_normalization_76/ReadVariableOp_1ReadVariableOp>sequential_19_batch_normalization_76_readvariableop_1_resource*
_output_shapes	
:А*
dtype027
5sequential_19/batch_normalization_76/ReadVariableOp_1Ч
Dsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_19_batch_normalization_76_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02F
Dsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOpЭ
Fsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_19_batch_normalization_76_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02H
Fsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1т
5sequential_19/batch_normalization_76/FusedBatchNormV3FusedBatchNormV3;sequential_19/conv2d_transpose_27/conv2d_transpose:output:0;sequential_19/batch_normalization_76/ReadVariableOp:value:0=sequential_19/batch_normalization_76/ReadVariableOp_1:value:0Lsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 27
5sequential_19/batch_normalization_76/FusedBatchNormV3Э
*sequential_19/batch_normalization_76/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2,
*sequential_19/batch_normalization_76/Const╩
&sequential_19/leaky_re_lu_73/LeakyRelu	LeakyRelu9sequential_19/batch_normalization_76/FusedBatchNormV3:y:0*0
_output_shapes
:         А2(
&sequential_19/leaky_re_lu_73/LeakyRelu╢
'sequential_19/conv2d_transpose_28/ShapeShape4sequential_19/leaky_re_lu_73/LeakyRelu:activations:0*
T0*
_output_shapes
:2)
'sequential_19/conv2d_transpose_28/Shape╕
5sequential_19/conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_19/conv2d_transpose_28/strided_slice/stack╝
7sequential_19/conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_28/strided_slice/stack_1╝
7sequential_19/conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_28/strided_slice/stack_2о
/sequential_19/conv2d_transpose_28/strided_sliceStridedSlice0sequential_19/conv2d_transpose_28/Shape:output:0>sequential_19/conv2d_transpose_28/strided_slice/stack:output:0@sequential_19/conv2d_transpose_28/strided_slice/stack_1:output:0@sequential_19/conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_19/conv2d_transpose_28/strided_slice╝
7sequential_19/conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_28/strided_slice_1/stack└
9sequential_19/conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_28/strided_slice_1/stack_1└
9sequential_19/conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_28/strided_slice_1/stack_2╕
1sequential_19/conv2d_transpose_28/strided_slice_1StridedSlice0sequential_19/conv2d_transpose_28/Shape:output:0@sequential_19/conv2d_transpose_28/strided_slice_1/stack:output:0Bsequential_19/conv2d_transpose_28/strided_slice_1/stack_1:output:0Bsequential_19/conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_19/conv2d_transpose_28/strided_slice_1╝
7sequential_19/conv2d_transpose_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_28/strided_slice_2/stack└
9sequential_19/conv2d_transpose_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_28/strided_slice_2/stack_1└
9sequential_19/conv2d_transpose_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_28/strided_slice_2/stack_2╕
1sequential_19/conv2d_transpose_28/strided_slice_2StridedSlice0sequential_19/conv2d_transpose_28/Shape:output:0@sequential_19/conv2d_transpose_28/strided_slice_2/stack:output:0Bsequential_19/conv2d_transpose_28/strided_slice_2/stack_1:output:0Bsequential_19/conv2d_transpose_28/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_19/conv2d_transpose_28/strided_slice_2Ф
'sequential_19/conv2d_transpose_28/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_19/conv2d_transpose_28/mul/yф
%sequential_19/conv2d_transpose_28/mulMul:sequential_19/conv2d_transpose_28/strided_slice_1:output:00sequential_19/conv2d_transpose_28/mul/y:output:0*
T0*
_output_shapes
: 2'
%sequential_19/conv2d_transpose_28/mulШ
)sequential_19/conv2d_transpose_28/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_19/conv2d_transpose_28/mul_1/yъ
'sequential_19/conv2d_transpose_28/mul_1Mul:sequential_19/conv2d_transpose_28/strided_slice_2:output:02sequential_19/conv2d_transpose_28/mul_1/y:output:0*
T0*
_output_shapes
: 2)
'sequential_19/conv2d_transpose_28/mul_1Ш
)sequential_19/conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2+
)sequential_19/conv2d_transpose_28/stack/3╬
'sequential_19/conv2d_transpose_28/stackPack8sequential_19/conv2d_transpose_28/strided_slice:output:0)sequential_19/conv2d_transpose_28/mul:z:0+sequential_19/conv2d_transpose_28/mul_1:z:02sequential_19/conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_19/conv2d_transpose_28/stack╝
7sequential_19/conv2d_transpose_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_19/conv2d_transpose_28/strided_slice_3/stack└
9sequential_19/conv2d_transpose_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_28/strided_slice_3/stack_1└
9sequential_19/conv2d_transpose_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_28/strided_slice_3/stack_2╕
1sequential_19/conv2d_transpose_28/strided_slice_3StridedSlice0sequential_19/conv2d_transpose_28/stack:output:0@sequential_19/conv2d_transpose_28/strided_slice_3/stack:output:0Bsequential_19/conv2d_transpose_28/strided_slice_3/stack_1:output:0Bsequential_19/conv2d_transpose_28/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_19/conv2d_transpose_28/strided_slice_3Ъ
Asequential_19/conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_19_conv2d_transpose_28_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype02C
Asequential_19/conv2d_transpose_28/conv2d_transpose/ReadVariableOpФ
2sequential_19/conv2d_transpose_28/conv2d_transposeConv2DBackpropInput0sequential_19/conv2d_transpose_28/stack:output:0Isequential_19/conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:04sequential_19/leaky_re_lu_73/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
24
2sequential_19/conv2d_transpose_28/conv2d_transposeи
1sequential_19/batch_normalization_77/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1sequential_19/batch_normalization_77/LogicalAnd/xи
1sequential_19/batch_normalization_77/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1sequential_19/batch_normalization_77/LogicalAnd/yА
/sequential_19/batch_normalization_77/LogicalAnd
LogicalAnd:sequential_19/batch_normalization_77/LogicalAnd/x:output:0:sequential_19/batch_normalization_77/LogicalAnd/y:output:0*
_output_shapes
: 21
/sequential_19/batch_normalization_77/LogicalAndу
3sequential_19/batch_normalization_77/ReadVariableOpReadVariableOp<sequential_19_batch_normalization_77_readvariableop_resource*
_output_shapes
:@*
dtype025
3sequential_19/batch_normalization_77/ReadVariableOpщ
5sequential_19/batch_normalization_77/ReadVariableOp_1ReadVariableOp>sequential_19_batch_normalization_77_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5sequential_19/batch_normalization_77/ReadVariableOp_1Ц
Dsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_19_batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_19_batch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Fsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1▌
5sequential_19/batch_normalization_77/FusedBatchNormV3FusedBatchNormV3;sequential_19/conv2d_transpose_28/conv2d_transpose:output:0;sequential_19/batch_normalization_77/ReadVariableOp:value:0=sequential_19/batch_normalization_77/ReadVariableOp_1:value:0Lsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 27
5sequential_19/batch_normalization_77/FusedBatchNormV3Э
*sequential_19/batch_normalization_77/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2,
*sequential_19/batch_normalization_77/Const╔
&sequential_19/leaky_re_lu_74/LeakyRelu	LeakyRelu9sequential_19/batch_normalization_77/FusedBatchNormV3:y:0*/
_output_shapes
:         @2(
&sequential_19/leaky_re_lu_74/LeakyRelu╢
'sequential_19/conv2d_transpose_29/ShapeShape4sequential_19/leaky_re_lu_74/LeakyRelu:activations:0*
T0*
_output_shapes
:2)
'sequential_19/conv2d_transpose_29/Shape╕
5sequential_19/conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_19/conv2d_transpose_29/strided_slice/stack╝
7sequential_19/conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_29/strided_slice/stack_1╝
7sequential_19/conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_29/strided_slice/stack_2о
/sequential_19/conv2d_transpose_29/strided_sliceStridedSlice0sequential_19/conv2d_transpose_29/Shape:output:0>sequential_19/conv2d_transpose_29/strided_slice/stack:output:0@sequential_19/conv2d_transpose_29/strided_slice/stack_1:output:0@sequential_19/conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_19/conv2d_transpose_29/strided_slice╝
7sequential_19/conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_29/strided_slice_1/stack└
9sequential_19/conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_29/strided_slice_1/stack_1└
9sequential_19/conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_29/strided_slice_1/stack_2╕
1sequential_19/conv2d_transpose_29/strided_slice_1StridedSlice0sequential_19/conv2d_transpose_29/Shape:output:0@sequential_19/conv2d_transpose_29/strided_slice_1/stack:output:0Bsequential_19/conv2d_transpose_29/strided_slice_1/stack_1:output:0Bsequential_19/conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_19/conv2d_transpose_29/strided_slice_1╝
7sequential_19/conv2d_transpose_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_19/conv2d_transpose_29/strided_slice_2/stack└
9sequential_19/conv2d_transpose_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_29/strided_slice_2/stack_1└
9sequential_19/conv2d_transpose_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_29/strided_slice_2/stack_2╕
1sequential_19/conv2d_transpose_29/strided_slice_2StridedSlice0sequential_19/conv2d_transpose_29/Shape:output:0@sequential_19/conv2d_transpose_29/strided_slice_2/stack:output:0Bsequential_19/conv2d_transpose_29/strided_slice_2/stack_1:output:0Bsequential_19/conv2d_transpose_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_19/conv2d_transpose_29/strided_slice_2Ф
'sequential_19/conv2d_transpose_29/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_19/conv2d_transpose_29/mul/yф
%sequential_19/conv2d_transpose_29/mulMul:sequential_19/conv2d_transpose_29/strided_slice_1:output:00sequential_19/conv2d_transpose_29/mul/y:output:0*
T0*
_output_shapes
: 2'
%sequential_19/conv2d_transpose_29/mulШ
)sequential_19/conv2d_transpose_29/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_19/conv2d_transpose_29/mul_1/yъ
'sequential_19/conv2d_transpose_29/mul_1Mul:sequential_19/conv2d_transpose_29/strided_slice_2:output:02sequential_19/conv2d_transpose_29/mul_1/y:output:0*
T0*
_output_shapes
: 2)
'sequential_19/conv2d_transpose_29/mul_1Ш
)sequential_19/conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_19/conv2d_transpose_29/stack/3╬
'sequential_19/conv2d_transpose_29/stackPack8sequential_19/conv2d_transpose_29/strided_slice:output:0)sequential_19/conv2d_transpose_29/mul:z:0+sequential_19/conv2d_transpose_29/mul_1:z:02sequential_19/conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_19/conv2d_transpose_29/stack╝
7sequential_19/conv2d_transpose_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_19/conv2d_transpose_29/strided_slice_3/stack└
9sequential_19/conv2d_transpose_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_29/strided_slice_3/stack_1└
9sequential_19/conv2d_transpose_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_19/conv2d_transpose_29/strided_slice_3/stack_2╕
1sequential_19/conv2d_transpose_29/strided_slice_3StridedSlice0sequential_19/conv2d_transpose_29/stack:output:0@sequential_19/conv2d_transpose_29/strided_slice_3/stack:output:0Bsequential_19/conv2d_transpose_29/strided_slice_3/stack_1:output:0Bsequential_19/conv2d_transpose_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_19/conv2d_transpose_29/strided_slice_3Щ
Asequential_19/conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_19_conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02C
Asequential_19/conv2d_transpose_29/conv2d_transpose/ReadVariableOpФ
2sequential_19/conv2d_transpose_29/conv2d_transposeConv2DBackpropInput0sequential_19/conv2d_transpose_29/stack:output:0Isequential_19/conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:04sequential_19/leaky_re_lu_74/LeakyRelu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
24
2sequential_19/conv2d_transpose_29/conv2d_transpose╧
&sequential_19/conv2d_transpose_29/TanhTanh;sequential_19/conv2d_transpose_29/conv2d_transpose:output:0*
T0*/
_output_shapes
:         2(
&sequential_19/conv2d_transpose_29/TanhЬ
IdentityIdentity*sequential_19/conv2d_transpose_29/Tanh:y:0>^sequential_19/batch_normalization_74/batchnorm/ReadVariableOp@^sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_1@^sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_2B^sequential_19/batch_normalization_74/batchnorm/mul/ReadVariableOp>^sequential_19/batch_normalization_75/batchnorm/ReadVariableOp@^sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_1@^sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_2B^sequential_19/batch_normalization_75/batchnorm/mul/ReadVariableOpE^sequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOpG^sequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_14^sequential_19/batch_normalization_76/ReadVariableOp6^sequential_19/batch_normalization_76/ReadVariableOp_1E^sequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOpG^sequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_14^sequential_19/batch_normalization_77/ReadVariableOp6^sequential_19/batch_normalization_77/ReadVariableOp_1B^sequential_19/conv2d_transpose_27/conv2d_transpose/ReadVariableOpB^sequential_19/conv2d_transpose_28/conv2d_transpose/ReadVariableOpB^sequential_19/conv2d_transpose_29/conv2d_transpose/ReadVariableOp.^sequential_19/dense_33/BiasAdd/ReadVariableOp-^sequential_19/dense_33/MatMul/ReadVariableOp.^sequential_19/dense_34/BiasAdd/ReadVariableOp-^sequential_19/dense_34/MatMul/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::2~
=sequential_19/batch_normalization_74/batchnorm/ReadVariableOp=sequential_19/batch_normalization_74/batchnorm/ReadVariableOp2В
?sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_1?sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_12В
?sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_2?sequential_19/batch_normalization_74/batchnorm/ReadVariableOp_22Ж
Asequential_19/batch_normalization_74/batchnorm/mul/ReadVariableOpAsequential_19/batch_normalization_74/batchnorm/mul/ReadVariableOp2~
=sequential_19/batch_normalization_75/batchnorm/ReadVariableOp=sequential_19/batch_normalization_75/batchnorm/ReadVariableOp2В
?sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_1?sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_12В
?sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_2?sequential_19/batch_normalization_75/batchnorm/ReadVariableOp_22Ж
Asequential_19/batch_normalization_75/batchnorm/mul/ReadVariableOpAsequential_19/batch_normalization_75/batchnorm/mul/ReadVariableOp2М
Dsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOpDsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1Fsequential_19/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_12j
3sequential_19/batch_normalization_76/ReadVariableOp3sequential_19/batch_normalization_76/ReadVariableOp2n
5sequential_19/batch_normalization_76/ReadVariableOp_15sequential_19/batch_normalization_76/ReadVariableOp_12М
Dsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOpDsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1Fsequential_19/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12j
3sequential_19/batch_normalization_77/ReadVariableOp3sequential_19/batch_normalization_77/ReadVariableOp2n
5sequential_19/batch_normalization_77/ReadVariableOp_15sequential_19/batch_normalization_77/ReadVariableOp_12Ж
Asequential_19/conv2d_transpose_27/conv2d_transpose/ReadVariableOpAsequential_19/conv2d_transpose_27/conv2d_transpose/ReadVariableOp2Ж
Asequential_19/conv2d_transpose_28/conv2d_transpose/ReadVariableOpAsequential_19/conv2d_transpose_28/conv2d_transpose/ReadVariableOp2Ж
Asequential_19/conv2d_transpose_29/conv2d_transpose/ReadVariableOpAsequential_19/conv2d_transpose_29/conv2d_transpose/ReadVariableOp2^
-sequential_19/dense_33/BiasAdd/ReadVariableOp-sequential_19/dense_33/BiasAdd/ReadVariableOp2\
,sequential_19/dense_33/MatMul/ReadVariableOp,sequential_19/dense_33/MatMul/ReadVariableOp2^
-sequential_19/dense_34/BiasAdd/ReadVariableOp-sequential_19/dense_34/BiasAdd/ReadVariableOp2\
,sequential_19/dense_34/MatMul/ReadVariableOp,sequential_19/dense_34/MatMul/ReadVariableOp:. *
(
_user_specified_namedense_33_input
ч
Й
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_374812

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOp^
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

LogicalAndУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
┤
K
/__inference_leaky_re_lu_73_layer_call_fn_376747

inputs
identity╨
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_73_layer_call_and_return_conditional_losses_3755142
PartitionedCallЗ
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:& "
 
_user_specified_nameinputs
ч
Й
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_374956

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOp^
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

LogicalAndУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Аb2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аb2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Аb2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Аb*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аb2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аb2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         Аb::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
Н
ї
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_376719

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constэ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Н
ї
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_375126

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constэ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
╗$
Э
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_376793

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_376778
assignmovingavg_1_376785
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/376778*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xп
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/376778*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_376778*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp╠
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/376778*
_output_shapes
:@2
AssignMovingAvg/sub_1╡
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/376778*
_output_shapes
:@2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_376778AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/376778*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/376785*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╖
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/376785*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_376785*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╪
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/376785*
_output_shapes
:@2
AssignMovingAvg_1/sub_1┐
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/376785*
_output_shapes
:@2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_376785AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/376785*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╕
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
▄ 
╛
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_375335

inputs,
(conv2d_transpose_readvariableop_resource
identityИвconv2d_transpose/ReadVariableOpD
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
strided_slice/stack_2т
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
strided_slice_1/stack_2ь
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
strided_slice_2/stack_2ь
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
stack/3В
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
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOpЁ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
conv2d_transpose{
TanhTanhconv2d_transpose:output:0*
T0*A
_output_shapes/
-:+                           2
TanhШ
IdentityIdentityTanh:y:0 ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+                           @:2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
Ё
▌
D__inference_dense_34_layer_call_and_return_conditional_losses_375414

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ААb*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аb2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Аb*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аb2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
В
ї
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_375296

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
═
▒
.__inference_sequential_19_layer_call_fn_376346

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
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_3757172
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
┴
a
E__inference_reshape_9_layer_call_and_return_conditional_losses_376646

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
strided_slice/stack_2т
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
B :А2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:         А2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Аb:& "
 
_user_specified_nameinputs
щ
f
J__inference_leaky_re_lu_74_layer_call_and_return_conditional_losses_376838

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           @2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:& "
 
_user_specified_nameinputs
яd
Э
"__inference__traced_restore_377017
file_prefix$
 assignvariableop_dense_33_kernel$
 assignvariableop_1_dense_33_bias3
/assignvariableop_2_batch_normalization_74_gamma2
.assignvariableop_3_batch_normalization_74_beta9
5assignvariableop_4_batch_normalization_74_moving_mean=
9assignvariableop_5_batch_normalization_74_moving_variance&
"assignvariableop_6_dense_34_kernel$
 assignvariableop_7_dense_34_bias3
/assignvariableop_8_batch_normalization_75_gamma2
.assignvariableop_9_batch_normalization_75_beta:
6assignvariableop_10_batch_normalization_75_moving_mean>
:assignvariableop_11_batch_normalization_75_moving_variance2
.assignvariableop_12_conv2d_transpose_27_kernel4
0assignvariableop_13_batch_normalization_76_gamma3
/assignvariableop_14_batch_normalization_76_beta:
6assignvariableop_15_batch_normalization_76_moving_mean>
:assignvariableop_16_batch_normalization_76_moving_variance2
.assignvariableop_17_conv2d_transpose_28_kernel4
0assignvariableop_18_batch_normalization_77_gamma3
/assignvariableop_19_batch_normalization_77_beta:
6assignvariableop_20_batch_normalization_77_moving_mean>
:assignvariableop_21_batch_normalization_77_moving_variance2
.assignvariableop_22_conv2d_transpose_29_kernel
identity_24ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1╣
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*┼

value╗
B╕
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names╝
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЮ
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

IdentityР
AssignVariableOpAssignVariableOp assignvariableop_dense_33_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_33_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2е
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_74_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3д
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_74_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4л
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_74_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5п
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_74_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ш
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_34_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ц
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_34_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8е
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_75_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9д
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_75_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10п
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_75_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11│
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_75_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12з
AssignVariableOp_12AssignVariableOp.assignvariableop_12_conv2d_transpose_27_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13й
AssignVariableOp_13AssignVariableOp0assignvariableop_13_batch_normalization_76_gammaIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14и
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_76_betaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15п
AssignVariableOp_15AssignVariableOp6assignvariableop_15_batch_normalization_76_moving_meanIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16│
AssignVariableOp_16AssignVariableOp:assignvariableop_16_batch_normalization_76_moving_varianceIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17з
AssignVariableOp_17AssignVariableOp.assignvariableop_17_conv2d_transpose_28_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18й
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_77_gammaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19и
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_77_betaIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20п
AssignVariableOp_20AssignVariableOp6assignvariableop_20_batch_normalization_77_moving_meanIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21│
AssignVariableOp_21AssignVariableOp:assignvariableop_21_batch_normalization_77_moving_varianceIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22з
AssignVariableOp_22AssignVariableOp.assignvariableop_22_conv2d_transpose_29_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
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
NoOp╪
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23х
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
╦Q
Р
I__inference_sequential_19_layer_call_and_return_conditional_losses_375563
dense_33_input+
'dense_33_statefulpartitionedcall_args_1+
'dense_33_statefulpartitionedcall_args_29
5batch_normalization_74_statefulpartitionedcall_args_19
5batch_normalization_74_statefulpartitionedcall_args_29
5batch_normalization_74_statefulpartitionedcall_args_39
5batch_normalization_74_statefulpartitionedcall_args_4+
'dense_34_statefulpartitionedcall_args_1+
'dense_34_statefulpartitionedcall_args_29
5batch_normalization_75_statefulpartitionedcall_args_19
5batch_normalization_75_statefulpartitionedcall_args_29
5batch_normalization_75_statefulpartitionedcall_args_39
5batch_normalization_75_statefulpartitionedcall_args_46
2conv2d_transpose_27_statefulpartitionedcall_args_19
5batch_normalization_76_statefulpartitionedcall_args_19
5batch_normalization_76_statefulpartitionedcall_args_29
5batch_normalization_76_statefulpartitionedcall_args_39
5batch_normalization_76_statefulpartitionedcall_args_46
2conv2d_transpose_28_statefulpartitionedcall_args_19
5batch_normalization_77_statefulpartitionedcall_args_19
5batch_normalization_77_statefulpartitionedcall_args_29
5batch_normalization_77_statefulpartitionedcall_args_39
5batch_normalization_77_statefulpartitionedcall_args_46
2conv2d_transpose_29_statefulpartitionedcall_args_1
identityИв.batch_normalization_74/StatefulPartitionedCallв.batch_normalization_75/StatefulPartitionedCallв.batch_normalization_76/StatefulPartitionedCallв.batch_normalization_77/StatefulPartitionedCallв+conv2d_transpose_27/StatefulPartitionedCallв+conv2d_transpose_28/StatefulPartitionedCallв+conv2d_transpose_29/StatefulPartitionedCallв dense_33/StatefulPartitionedCallв dense_34/StatefulPartitionedCall╢
 dense_33/StatefulPartitionedCallStatefulPartitionedCalldense_33_input'dense_33_statefulpartitionedcall_args_1'dense_33_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_3753562"
 dense_33/StatefulPartitionedCallЗ
.batch_normalization_74/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:05batch_normalization_74_statefulpartitionedcall_args_15batch_normalization_74_statefulpartitionedcall_args_25batch_normalization_74_statefulpartitionedcall_args_35batch_normalization_74_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_37478020
.batch_normalization_74/StatefulPartitionedCallЕ
leaky_re_lu_71/PartitionedCallPartitionedCall7batch_normalization_74/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_3753962 
leaky_re_lu_71/PartitionedCall╧
 dense_34/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0'dense_34_statefulpartitionedcall_args_1'dense_34_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3754142"
 dense_34/StatefulPartitionedCallЗ
.batch_normalization_75/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:05batch_normalization_75_statefulpartitionedcall_args_15batch_normalization_75_statefulpartitionedcall_args_25batch_normalization_75_statefulpartitionedcall_args_35batch_normalization_75_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_37492420
.batch_normalization_75/StatefulPartitionedCallЕ
leaky_re_lu_72/PartitionedCallPartitionedCall7batch_normalization_75/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_72_layer_call_and_return_conditional_losses_3754542 
leaky_re_lu_72/PartitionedCallю
reshape_9/PartitionedCallPartitionedCall'leaky_re_lu_72/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_reshape_9_layer_call_and_return_conditional_losses_3754762
reshape_9/PartitionedCallц
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall"reshape_9/PartitionedCall:output:02conv2d_transpose_27_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3749942-
+conv2d_transpose_27/StatefulPartitionedCallм
.batch_normalization_76/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:05batch_normalization_76_statefulpartitionedcall_args_15batch_normalization_76_statefulpartitionedcall_args_25batch_normalization_76_statefulpartitionedcall_args_35batch_normalization_76_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_37509520
.batch_normalization_76/StatefulPartitionedCallЯ
leaky_re_lu_73/PartitionedCallPartitionedCall7batch_normalization_76/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_73_layer_call_and_return_conditional_losses_3755142 
leaky_re_lu_73/PartitionedCallъ
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_73/PartitionedCall:output:02conv2d_transpose_28_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3751642-
+conv2d_transpose_28/StatefulPartitionedCallл
.batch_normalization_77/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:05batch_normalization_77_statefulpartitionedcall_args_15batch_normalization_77_statefulpartitionedcall_args_25batch_normalization_77_statefulpartitionedcall_args_35batch_normalization_77_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_37526520
.batch_normalization_77/StatefulPartitionedCallЮ
leaky_re_lu_74/PartitionedCallPartitionedCall7batch_normalization_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_74_layer_call_and_return_conditional_losses_3755522 
leaky_re_lu_74/PartitionedCallъ
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_74/PartitionedCall:output:02conv2d_transpose_29_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3753352-
+conv2d_transpose_29/StatefulPartitionedCall╢
IdentityIdentity4conv2d_transpose_29/StatefulPartitionedCall:output:0/^batch_normalization_74/StatefulPartitionedCall/^batch_normalization_75/StatefulPartitionedCall/^batch_normalization_76/StatefulPartitionedCall/^batch_normalization_77/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::2`
.batch_normalization_74/StatefulPartitionedCall.batch_normalization_74/StatefulPartitionedCall2`
.batch_normalization_75/StatefulPartitionedCall.batch_normalization_75/StatefulPartitionedCall2`
.batch_normalization_76/StatefulPartitionedCall.batch_normalization_76/StatefulPartitionedCall2`
.batch_normalization_77/StatefulPartitionedCall.batch_normalization_77/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:. *
(
_user_specified_namedense_33_input
│Q
И
I__inference_sequential_19_layer_call_and_return_conditional_losses_375648

inputs+
'dense_33_statefulpartitionedcall_args_1+
'dense_33_statefulpartitionedcall_args_29
5batch_normalization_74_statefulpartitionedcall_args_19
5batch_normalization_74_statefulpartitionedcall_args_29
5batch_normalization_74_statefulpartitionedcall_args_39
5batch_normalization_74_statefulpartitionedcall_args_4+
'dense_34_statefulpartitionedcall_args_1+
'dense_34_statefulpartitionedcall_args_29
5batch_normalization_75_statefulpartitionedcall_args_19
5batch_normalization_75_statefulpartitionedcall_args_29
5batch_normalization_75_statefulpartitionedcall_args_39
5batch_normalization_75_statefulpartitionedcall_args_46
2conv2d_transpose_27_statefulpartitionedcall_args_19
5batch_normalization_76_statefulpartitionedcall_args_19
5batch_normalization_76_statefulpartitionedcall_args_29
5batch_normalization_76_statefulpartitionedcall_args_39
5batch_normalization_76_statefulpartitionedcall_args_46
2conv2d_transpose_28_statefulpartitionedcall_args_19
5batch_normalization_77_statefulpartitionedcall_args_19
5batch_normalization_77_statefulpartitionedcall_args_29
5batch_normalization_77_statefulpartitionedcall_args_39
5batch_normalization_77_statefulpartitionedcall_args_46
2conv2d_transpose_29_statefulpartitionedcall_args_1
identityИв.batch_normalization_74/StatefulPartitionedCallв.batch_normalization_75/StatefulPartitionedCallв.batch_normalization_76/StatefulPartitionedCallв.batch_normalization_77/StatefulPartitionedCallв+conv2d_transpose_27/StatefulPartitionedCallв+conv2d_transpose_28/StatefulPartitionedCallв+conv2d_transpose_29/StatefulPartitionedCallв dense_33/StatefulPartitionedCallв dense_34/StatefulPartitionedCallо
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinputs'dense_33_statefulpartitionedcall_args_1'dense_33_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_3753562"
 dense_33/StatefulPartitionedCallЗ
.batch_normalization_74/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:05batch_normalization_74_statefulpartitionedcall_args_15batch_normalization_74_statefulpartitionedcall_args_25batch_normalization_74_statefulpartitionedcall_args_35batch_normalization_74_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_37478020
.batch_normalization_74/StatefulPartitionedCallЕ
leaky_re_lu_71/PartitionedCallPartitionedCall7batch_normalization_74/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_3753962 
leaky_re_lu_71/PartitionedCall╧
 dense_34/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0'dense_34_statefulpartitionedcall_args_1'dense_34_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3754142"
 dense_34/StatefulPartitionedCallЗ
.batch_normalization_75/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:05batch_normalization_75_statefulpartitionedcall_args_15batch_normalization_75_statefulpartitionedcall_args_25batch_normalization_75_statefulpartitionedcall_args_35batch_normalization_75_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_37492420
.batch_normalization_75/StatefulPartitionedCallЕ
leaky_re_lu_72/PartitionedCallPartitionedCall7batch_normalization_75/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_72_layer_call_and_return_conditional_losses_3754542 
leaky_re_lu_72/PartitionedCallю
reshape_9/PartitionedCallPartitionedCall'leaky_re_lu_72/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_reshape_9_layer_call_and_return_conditional_losses_3754762
reshape_9/PartitionedCallц
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall"reshape_9/PartitionedCall:output:02conv2d_transpose_27_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3749942-
+conv2d_transpose_27/StatefulPartitionedCallм
.batch_normalization_76/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:05batch_normalization_76_statefulpartitionedcall_args_15batch_normalization_76_statefulpartitionedcall_args_25batch_normalization_76_statefulpartitionedcall_args_35batch_normalization_76_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_37509520
.batch_normalization_76/StatefulPartitionedCallЯ
leaky_re_lu_73/PartitionedCallPartitionedCall7batch_normalization_76/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_73_layer_call_and_return_conditional_losses_3755142 
leaky_re_lu_73/PartitionedCallъ
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_73/PartitionedCall:output:02conv2d_transpose_28_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3751642-
+conv2d_transpose_28/StatefulPartitionedCallл
.batch_normalization_77/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:05batch_normalization_77_statefulpartitionedcall_args_15batch_normalization_77_statefulpartitionedcall_args_25batch_normalization_77_statefulpartitionedcall_args_35batch_normalization_77_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_37526520
.batch_normalization_77/StatefulPartitionedCallЮ
leaky_re_lu_74/PartitionedCallPartitionedCall7batch_normalization_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_74_layer_call_and_return_conditional_losses_3755522 
leaky_re_lu_74/PartitionedCallъ
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_74/PartitionedCall:output:02conv2d_transpose_29_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3753352-
+conv2d_transpose_29/StatefulPartitionedCall╢
IdentityIdentity4conv2d_transpose_29/StatefulPartitionedCall:output:0/^batch_normalization_74/StatefulPartitionedCall/^batch_normalization_75/StatefulPartitionedCall/^batch_normalization_76/StatefulPartitionedCall/^batch_normalization_77/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*В
_input_shapesq
o:         d:::::::::::::::::::::::2`
.batch_normalization_74/StatefulPartitionedCall.batch_normalization_74/StatefulPartitionedCall2`
.batch_normalization_75/StatefulPartitionedCall.batch_normalization_75/StatefulPartitionedCall2`
.batch_normalization_76/StatefulPartitionedCall.batch_normalization_76/StatefulPartitionedCall2`
.batch_normalization_77/StatefulPartitionedCall.batch_normalization_77/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
°
к
)__inference_dense_34_layer_call_fn_376506

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         Аb*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3754142
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         Аb2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
┤/
╔
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_374780

inputs
assignmovingavg_374755
assignmovingavg_1_374761)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOp^
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

LogicalAndК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/374755*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_374755*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/374755*
_output_shapes	
:А2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/374755*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_374755AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/374755*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/374761*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_374761*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/374761*
_output_shapes	
:А2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/374761*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_374761AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/374761*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╠
serving_default╕
I
dense_33_input7
 serving_default_dense_33_input:0         dO
conv2d_transpose_298
StatefulPartitionedCall:0         tensorflow/serving/predict:ШФ
║`
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
д__call__
е_default_save_signature
+ж&call_and_return_all_conditional_losses"╔[
_tf_keras_sequentialк[{"class_name": "Sequential", "name": "sequential_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_19", "layers": [{"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "batch_input_shape": [null, 100], "dtype": "float32", "units": 1024, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_74", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_71", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_75", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_72", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Reshape", "config": {"name": "reshape_9", "trainable": true, "dtype": "float32", "target_shape": [7, 7, 256]}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_27", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_76", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_73", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_28", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_77", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_74", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_29", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_19", "layers": [{"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "batch_input_shape": [null, 100], "dtype": "float32", "units": 1024, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_74", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_71", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_75", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_72", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Reshape", "config": {"name": "reshape_9", "trainable": true, "dtype": "float32", "target_shape": [7, 7, 256]}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_27", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_76", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_73", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_28", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_77", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_74", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_29", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}}}
п"м
_tf_keras_input_layerМ{"class_name": "InputLayer", "name": "dense_33_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 100], "config": {"batch_input_shape": [null, 100], "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_33_input"}}
г

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
з__call__
+и&call_and_return_all_conditional_losses"№
_tf_keras_layerт{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 100], "config": {"name": "dense_33", "trainable": true, "batch_input_shape": [null, 100], "dtype": "float32", "units": 1024, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
╣
axis
	gamma
beta
moving_mean
moving_variance
 	variables
!regularization_losses
"trainable_variables
#	keras_api
й__call__
+к&call_and_return_all_conditional_losses"у
_tf_keras_layer╔{"class_name": "BatchNormalization", "name": "batch_normalization_74", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_74", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 1024}}}}
м
$	variables
%regularization_losses
&trainable_variables
'	keras_api
л__call__
+м&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "LeakyReLU", "name": "leaky_re_lu_71", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_71", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
№

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
н__call__
+о&call_and_return_all_conditional_losses"╒
_tf_keras_layer╗{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}}
║
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4regularization_losses
5trainable_variables
6	keras_api
п__call__
+░&call_and_return_all_conditional_losses"ф
_tf_keras_layer╩{"class_name": "BatchNormalization", "name": "batch_normalization_75", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_75", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 12544}}}}
м
7	variables
8regularization_losses
9trainable_variables
:	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "LeakyReLU", "name": "leaky_re_lu_72", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_72", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Я
;	variables
<regularization_losses
=trainable_variables
>	keras_api
│__call__
+┤&call_and_return_all_conditional_losses"О
_tf_keras_layerЇ{"class_name": "Reshape", "name": "reshape_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "reshape_9", "trainable": true, "dtype": "float32", "target_shape": [7, 7, 256]}}
а

?kernel
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
╡__call__
+╢&call_and_return_all_conditional_losses"Г
_tf_keras_layerщ{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_27", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
╕
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
╖__call__
+╕&call_and_return_all_conditional_losses"т
_tf_keras_layer╚{"class_name": "BatchNormalization", "name": "batch_normalization_76", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_76", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
м
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
╣__call__
+║&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "LeakyReLU", "name": "leaky_re_lu_73", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_73", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Я

Qkernel
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
╗__call__
+╝&call_and_return_all_conditional_losses"В
_tf_keras_layerш{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_28", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
╖
Vaxis
	Wgamma
Xbeta
Ymoving_mean
Zmoving_variance
[	variables
\regularization_losses
]trainable_variables
^	keras_api
╜__call__
+╛&call_and_return_all_conditional_losses"с
_tf_keras_layer╟{"class_name": "BatchNormalization", "name": "batch_normalization_77", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_77", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
м
_	variables
`regularization_losses
atrainable_variables
b	keras_api
┐__call__
+└&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "LeakyReLU", "name": "leaky_re_lu_74", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_74", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ы

ckernel
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"■
_tf_keras_layerф{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_29", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
╬
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
О
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
╗
	variables
hmetrics
trainable_variables

ilayers
regularization_losses
jnon_trainable_variables
klayer_regularization_losses
д__call__
е_default_save_signature
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
-
├serving_default"
signature_map
": 	dА2dense_33/kernel
:А2dense_33/bias
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
Э
	variables
lmetrics
regularization_losses

mlayers
trainable_variables
nnon_trainable_variables
olayer_regularization_losses
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)А2batch_normalization_74/gamma
*:(А2batch_normalization_74/beta
3:1А (2"batch_normalization_74/moving_mean
7:5А (2&batch_normalization_74/moving_variance
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
Э
 	variables
pmetrics
!regularization_losses

qlayers
"trainable_variables
rnon_trainable_variables
slayer_regularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
$	variables
tmetrics
%regularization_losses

ulayers
&trainable_variables
vnon_trainable_variables
wlayer_regularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
#:!
ААb2dense_34/kernel
:Аb2dense_34/bias
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
Э
*	variables
xmetrics
+regularization_losses

ylayers
,trainable_variables
znon_trainable_variables
{layer_regularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)Аb2batch_normalization_75/gamma
*:(Аb2batch_normalization_75/beta
3:1Аb (2"batch_normalization_75/moving_mean
7:5Аb (2&batch_normalization_75/moving_variance
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
Э
3	variables
|metrics
4regularization_losses

}layers
5trainable_variables
~non_trainable_variables
layer_regularization_losses
п__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
7	variables
Аmetrics
8regularization_losses
Бlayers
9trainable_variables
Вnon_trainable_variables
 Гlayer_regularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
;	variables
Дmetrics
<regularization_losses
Еlayers
=trainable_variables
Жnon_trainable_variables
 Зlayer_regularization_losses
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
6:4АА2conv2d_transpose_27/kernel
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
?0"
trackable_list_wrapper
б
@	variables
Иmetrics
Aregularization_losses
Йlayers
Btrainable_variables
Кnon_trainable_variables
 Лlayer_regularization_losses
╡__call__
+╢&call_and_return_all_conditional_losses
'╢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)А2batch_normalization_76/gamma
*:(А2batch_normalization_76/beta
3:1А (2"batch_normalization_76/moving_mean
7:5А (2&batch_normalization_76/moving_variance
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
б
I	variables
Мmetrics
Jregularization_losses
Нlayers
Ktrainable_variables
Оnon_trainable_variables
 Пlayer_regularization_losses
╖__call__
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
M	variables
Рmetrics
Nregularization_losses
Сlayers
Otrainable_variables
Тnon_trainable_variables
 Уlayer_regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
5:3@А2conv2d_transpose_28/kernel
'
Q0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
б
R	variables
Фmetrics
Sregularization_losses
Хlayers
Ttrainable_variables
Цnon_trainable_variables
 Чlayer_regularization_losses
╗__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_77/gamma
):'@2batch_normalization_77/beta
2:0@ (2"batch_normalization_77/moving_mean
6:4@ (2&batch_normalization_77/moving_variance
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
б
[	variables
Шmetrics
\regularization_losses
Щlayers
]trainable_variables
Ъnon_trainable_variables
 Ыlayer_regularization_losses
╜__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
_	variables
Ьmetrics
`regularization_losses
Эlayers
atrainable_variables
Юnon_trainable_variables
 Яlayer_regularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
4:2@2conv2d_transpose_29/kernel
'
c0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
c0"
trackable_list_wrapper
б
d	variables
аmetrics
eregularization_losses
бlayers
ftrainable_variables
вnon_trainable_variables
 гlayer_regularization_losses
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Ж
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
Ж2Г
.__inference_sequential_19_layer_call_fn_376318
.__inference_sequential_19_layer_call_fn_375743
.__inference_sequential_19_layer_call_fn_376346
.__inference_sequential_19_layer_call_fn_375674└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ц2у
!__inference__wrapped_model_374675╜
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *-в*
(К%
dense_33_input         d
Є2я
I__inference_sequential_19_layer_call_and_return_conditional_losses_376107
I__inference_sequential_19_layer_call_and_return_conditional_losses_376290
I__inference_sequential_19_layer_call_and_return_conditional_losses_375563
I__inference_sequential_19_layer_call_and_return_conditional_losses_375604└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╙2╨
)__inference_dense_33_layer_call_fn_376363в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_dense_33_layer_call_and_return_conditional_losses_376356в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
м2й
7__inference_batch_normalization_74_layer_call_fn_376470
7__inference_batch_normalization_74_layer_call_fn_376479┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_376438
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_376461┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┘2╓
/__inference_leaky_re_lu_71_layer_call_fn_376489в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_376484в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_dense_34_layer_call_fn_376506в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_dense_34_layer_call_and_return_conditional_losses_376499в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
м2й
7__inference_batch_normalization_75_layer_call_fn_376613
7__inference_batch_normalization_75_layer_call_fn_376622┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_376604
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_376581┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┘2╓
/__inference_leaky_re_lu_72_layer_call_fn_376632в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_leaky_re_lu_72_layer_call_and_return_conditional_losses_376627в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_reshape_9_layer_call_fn_376651в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_reshape_9_layer_call_and_return_conditional_losses_376646в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ф2С
4__inference_conv2d_transpose_27_layer_call_fn_375001╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
п2м
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_374994╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
м2й
7__inference_batch_normalization_76_layer_call_fn_376737
7__inference_batch_normalization_76_layer_call_fn_376728┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_376719
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_376697┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┘2╓
/__inference_leaky_re_lu_73_layer_call_fn_376747в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_leaky_re_lu_73_layer_call_and_return_conditional_losses_376742в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ф2С
4__inference_conv2d_transpose_28_layer_call_fn_375171╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
п2м
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_375164╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
м2й
7__inference_batch_normalization_77_layer_call_fn_376824
7__inference_batch_normalization_77_layer_call_fn_376833┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_376815
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_376793┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┘2╓
/__inference_leaky_re_lu_74_layer_call_fn_376843в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_leaky_re_lu_74_layer_call_and_return_conditional_losses_376838в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
У2Р
4__inference_conv2d_transpose_29_layer_call_fn_375342╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
о2л
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_375335╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
:B8
$__inference_signature_wrapper_375868dense_33_input╦
!__inference__wrapped_model_374675е()2/10?EFGHQWXYZc7в4
-в*
(К%
dense_33_input         d
к "QкN
L
conv2d_transpose_295К2
conv2d_transpose_29         ║
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_376438d4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ ║
R__inference_batch_normalization_74_layer_call_and_return_conditional_losses_376461d4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ Т
7__inference_batch_normalization_74_layer_call_fn_376470W4в1
*в'
!К
inputs         А
p
к "К         АТ
7__inference_batch_normalization_74_layer_call_fn_376479W4в1
*в'
!К
inputs         А
p 
к "К         А║
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_376581d12/04в1
*в'
!К
inputs         Аb
p
к "&в#
К
0         Аb
Ъ ║
R__inference_batch_normalization_75_layer_call_and_return_conditional_losses_376604d2/104в1
*в'
!К
inputs         Аb
p 
к "&в#
К
0         Аb
Ъ Т
7__inference_batch_normalization_75_layer_call_fn_376613W12/04в1
*в'
!К
inputs         Аb
p
к "К         АbТ
7__inference_batch_normalization_75_layer_call_fn_376622W2/104в1
*в'
!К
inputs         Аb
p 
к "К         Аbя
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_376697ШEFGHNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ я
R__inference_batch_normalization_76_layer_call_and_return_conditional_losses_376719ШEFGHNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ ╟
7__inference_batch_normalization_76_layer_call_fn_376728ЛEFGHNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А╟
7__inference_batch_normalization_76_layer_call_fn_376737ЛEFGHNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           Аэ
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_376793ЦWXYZMвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ э
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_376815ЦWXYZMвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ ┼
7__inference_batch_normalization_77_layer_call_fn_376824ЙWXYZMвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @┼
7__inference_batch_normalization_77_layer_call_fn_376833ЙWXYZMвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @х
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_374994С?JвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╜
4__inference_conv2d_transpose_27_layer_call_fn_375001Д?JвG
@в=
;К8
inputs,                           А
к "3К0,                           Аф
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_375164РQJвG
@в=
;К8
inputs,                           А
к "?в<
5К2
0+                           @
Ъ ╝
4__inference_conv2d_transpose_28_layer_call_fn_375171ГQJвG
@в=
;К8
inputs,                           А
к "2К/+                           @у
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_375335ПcIвF
?в<
:К7
inputs+                           @
к "?в<
5К2
0+                           
Ъ ╗
4__inference_conv2d_transpose_29_layer_call_fn_375342ВcIвF
?в<
:К7
inputs+                           @
к "2К/+                           е
D__inference_dense_33_layer_call_and_return_conditional_losses_376356]/в,
%в"
 К
inputs         d
к "&в#
К
0         А
Ъ }
)__inference_dense_33_layer_call_fn_376363P/в,
%в"
 К
inputs         d
к "К         Аж
D__inference_dense_34_layer_call_and_return_conditional_losses_376499^()0в-
&в#
!К
inputs         А
к "&в#
К
0         Аb
Ъ ~
)__inference_dense_34_layer_call_fn_376506Q()0в-
&в#
!К
inputs         А
к "К         Аbи
J__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_376484Z0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ А
/__inference_leaky_re_lu_71_layer_call_fn_376489M0в-
&в#
!К
inputs         А
к "К         Аи
J__inference_leaky_re_lu_72_layer_call_and_return_conditional_losses_376627Z0в-
&в#
!К
inputs         Аb
к "&в#
К
0         Аb
Ъ А
/__inference_leaky_re_lu_72_layer_call_fn_376632M0в-
&в#
!К
inputs         Аb
к "К         Аb▌
J__inference_leaky_re_lu_73_layer_call_and_return_conditional_losses_376742ОJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╡
/__inference_leaky_re_lu_73_layer_call_fn_376747БJвG
@в=
;К8
inputs,                           А
к "3К0,                           А█
J__inference_leaky_re_lu_74_layer_call_and_return_conditional_losses_376838МIвF
?в<
:К7
inputs+                           @
к "?в<
5К2
0+                           @
Ъ ▓
/__inference_leaky_re_lu_74_layer_call_fn_376843IвF
?в<
:К7
inputs+                           @
к "2К/+                           @л
E__inference_reshape_9_layer_call_and_return_conditional_losses_376646b0в-
&в#
!К
inputs         Аb
к ".в+
$К!
0         А
Ъ Г
*__inference_reshape_9_layer_call_fn_376651U0в-
&в#
!К
inputs         Аb
к "!К         Ащ
I__inference_sequential_19_layer_call_and_return_conditional_losses_375563Ы()12/0?EFGHQWXYZc?в<
5в2
(К%
dense_33_input         d
p

 
к "?в<
5К2
0+                           
Ъ щ
I__inference_sequential_19_layer_call_and_return_conditional_losses_375604Ы()2/10?EFGHQWXYZc?в<
5в2
(К%
dense_33_input         d
p 

 
к "?в<
5К2
0+                           
Ъ ╧
I__inference_sequential_19_layer_call_and_return_conditional_losses_376107Б()12/0?EFGHQWXYZc7в4
-в*
 К
inputs         d
p

 
к "-в*
#К 
0         
Ъ ╧
I__inference_sequential_19_layer_call_and_return_conditional_losses_376290Б()2/10?EFGHQWXYZc7в4
-в*
 К
inputs         d
p 

 
к "-в*
#К 
0         
Ъ ┴
.__inference_sequential_19_layer_call_fn_375674О()12/0?EFGHQWXYZc?в<
5в2
(К%
dense_33_input         d
p

 
к "2К/+                           ┴
.__inference_sequential_19_layer_call_fn_375743О()2/10?EFGHQWXYZc?в<
5в2
(К%
dense_33_input         d
p 

 
к "2К/+                           ╣
.__inference_sequential_19_layer_call_fn_376318Ж()12/0?EFGHQWXYZc7в4
-в*
 К
inputs         d
p

 
к "2К/+                           ╣
.__inference_sequential_19_layer_call_fn_376346Ж()2/10?EFGHQWXYZc7в4
-в*
 К
inputs         d
p 

 
к "2К/+                           р
$__inference_signature_wrapper_375868╖()2/10?EFGHQWXYZcIвF
в 
?к<
:
dense_33_input(К%
dense_33_input         d"QкN
L
conv2d_transpose_295К2
conv2d_transpose_29         