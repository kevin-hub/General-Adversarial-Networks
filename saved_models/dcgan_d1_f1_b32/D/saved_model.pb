╒║
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
shapeshapeИ"serve*2.1.02v2.1.0-0-ge5bf8de4108Ы╠
а
sequential_24/conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name sequential_24/conv2d_36/kernel
Щ
2sequential_24/conv2d_36/kernel/Read/ReadVariableOpReadVariableOpsequential_24/conv2d_36/kernel*&
_output_shapes
:@*
dtype0
Р
sequential_24/conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namesequential_24/conv2d_36/bias
Й
0sequential_24/conv2d_36/bias/Read/ReadVariableOpReadVariableOpsequential_24/conv2d_36/bias*
_output_shapes
:@*
dtype0
м
*sequential_24/batch_normalization_94/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*sequential_24/batch_normalization_94/gamma
е
>sequential_24/batch_normalization_94/gamma/Read/ReadVariableOpReadVariableOp*sequential_24/batch_normalization_94/gamma*
_output_shapes
:@*
dtype0
к
)sequential_24/batch_normalization_94/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)sequential_24/batch_normalization_94/beta
г
=sequential_24/batch_normalization_94/beta/Read/ReadVariableOpReadVariableOp)sequential_24/batch_normalization_94/beta*
_output_shapes
:@*
dtype0
╕
0sequential_24/batch_normalization_94/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20sequential_24/batch_normalization_94/moving_mean
▒
Dsequential_24/batch_normalization_94/moving_mean/Read/ReadVariableOpReadVariableOp0sequential_24/batch_normalization_94/moving_mean*
_output_shapes
:@*
dtype0
└
4sequential_24/batch_normalization_94/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64sequential_24/batch_normalization_94/moving_variance
╣
Hsequential_24/batch_normalization_94/moving_variance/Read/ReadVariableOpReadVariableOp4sequential_24/batch_normalization_94/moving_variance*
_output_shapes
:@*
dtype0
а
sequential_24/conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*/
shared_name sequential_24/conv2d_37/kernel
Щ
2sequential_24/conv2d_37/kernel/Read/ReadVariableOpReadVariableOpsequential_24/conv2d_37/kernel*&
_output_shapes
:@@*
dtype0
Р
sequential_24/conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namesequential_24/conv2d_37/bias
Й
0sequential_24/conv2d_37/bias/Read/ReadVariableOpReadVariableOpsequential_24/conv2d_37/bias*
_output_shapes
:@*
dtype0
м
*sequential_24/batch_normalization_95/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*sequential_24/batch_normalization_95/gamma
е
>sequential_24/batch_normalization_95/gamma/Read/ReadVariableOpReadVariableOp*sequential_24/batch_normalization_95/gamma*
_output_shapes
:@*
dtype0
к
)sequential_24/batch_normalization_95/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)sequential_24/batch_normalization_95/beta
г
=sequential_24/batch_normalization_95/beta/Read/ReadVariableOpReadVariableOp)sequential_24/batch_normalization_95/beta*
_output_shapes
:@*
dtype0
╕
0sequential_24/batch_normalization_95/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20sequential_24/batch_normalization_95/moving_mean
▒
Dsequential_24/batch_normalization_95/moving_mean/Read/ReadVariableOpReadVariableOp0sequential_24/batch_normalization_95/moving_mean*
_output_shapes
:@*
dtype0
└
4sequential_24/batch_normalization_95/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64sequential_24/batch_normalization_95/moving_variance
╣
Hsequential_24/batch_normalization_95/moving_variance/Read/ReadVariableOpReadVariableOp4sequential_24/batch_normalization_95/moving_variance*
_output_shapes
:@*
dtype0
б
sequential_24/conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*/
shared_name sequential_24/conv2d_38/kernel
Ъ
2sequential_24/conv2d_38/kernel/Read/ReadVariableOpReadVariableOpsequential_24/conv2d_38/kernel*'
_output_shapes
:@А*
dtype0
С
sequential_24/conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namesequential_24/conv2d_38/bias
К
0sequential_24/conv2d_38/bias/Read/ReadVariableOpReadVariableOpsequential_24/conv2d_38/bias*
_output_shapes	
:А*
dtype0
н
*sequential_24/batch_normalization_96/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*;
shared_name,*sequential_24/batch_normalization_96/gamma
ж
>sequential_24/batch_normalization_96/gamma/Read/ReadVariableOpReadVariableOp*sequential_24/batch_normalization_96/gamma*
_output_shapes	
:А*
dtype0
л
)sequential_24/batch_normalization_96/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*:
shared_name+)sequential_24/batch_normalization_96/beta
д
=sequential_24/batch_normalization_96/beta/Read/ReadVariableOpReadVariableOp)sequential_24/batch_normalization_96/beta*
_output_shapes	
:А*
dtype0
╣
0sequential_24/batch_normalization_96/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*A
shared_name20sequential_24/batch_normalization_96/moving_mean
▓
Dsequential_24/batch_normalization_96/moving_mean/Read/ReadVariableOpReadVariableOp0sequential_24/batch_normalization_96/moving_mean*
_output_shapes	
:А*
dtype0
┴
4sequential_24/batch_normalization_96/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*E
shared_name64sequential_24/batch_normalization_96/moving_variance
║
Hsequential_24/batch_normalization_96/moving_variance/Read/ReadVariableOpReadVariableOp4sequential_24/batch_normalization_96/moving_variance*
_output_shapes	
:А*
dtype0
Ч
sequential_24/dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А1*.
shared_namesequential_24/dense_43/kernel
Р
1sequential_24/dense_43/kernel/Read/ReadVariableOpReadVariableOpsequential_24/dense_43/kernel*
_output_shapes
:	А1*
dtype0
О
sequential_24/dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namesequential_24/dense_43/bias
З
/sequential_24/dense_43/bias/Read/ReadVariableOpReadVariableOpsequential_24/dense_43/bias*
_output_shapes
:*
dtype0
м
*sequential_24/batch_normalization_97/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*sequential_24/batch_normalization_97/gamma
е
>sequential_24/batch_normalization_97/gamma/Read/ReadVariableOpReadVariableOp*sequential_24/batch_normalization_97/gamma*
_output_shapes
:*
dtype0
к
)sequential_24/batch_normalization_97/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)sequential_24/batch_normalization_97/beta
г
=sequential_24/batch_normalization_97/beta/Read/ReadVariableOpReadVariableOp)sequential_24/batch_normalization_97/beta*
_output_shapes
:*
dtype0
╕
0sequential_24/batch_normalization_97/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20sequential_24/batch_normalization_97/moving_mean
▒
Dsequential_24/batch_normalization_97/moving_mean/Read/ReadVariableOpReadVariableOp0sequential_24/batch_normalization_97/moving_mean*
_output_shapes
:*
dtype0
└
4sequential_24/batch_normalization_97/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64sequential_24/batch_normalization_97/moving_variance
╣
Hsequential_24/batch_normalization_97/moving_variance/Read/ReadVariableOpReadVariableOp4sequential_24/batch_normalization_97/moving_variance*
_output_shapes
:*
dtype0
Ц
sequential_24/dense_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namesequential_24/dense_44/kernel
П
1sequential_24/dense_44/kernel/Read/ReadVariableOpReadVariableOpsequential_24/dense_44/kernel*
_output_shapes

:*
dtype0
О
sequential_24/dense_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namesequential_24/dense_44/bias
З
/sequential_24/dense_44/bias/Read/ReadVariableOpReadVariableOpsequential_24/dense_44/bias*
_output_shapes
:*
dtype0

NoOpNoOp
▐H
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЩH
valueПHBМH BЕH
╘
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
	variables
trainable_variables
regularization_losses
	keras_api

signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
Ч
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(regularization_losses
)trainable_variables
*	keras_api
R
+	variables
,regularization_losses
-trainable_variables
.	keras_api
R
/	variables
0regularization_losses
1trainable_variables
2	keras_api
h

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
Ч
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>	variables
?regularization_losses
@trainable_variables
A	keras_api
R
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
R
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
Ч
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
R
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
R
]	variables
^regularization_losses
_trainable_variables
`	keras_api
R
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
h

ekernel
fbias
g	variables
hregularization_losses
itrainable_variables
j	keras_api
Ч
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
R
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
h

xkernel
ybias
z	variables
{regularization_losses
|trainable_variables
}	keras_api
╞
0
1
#2
$3
%4
&5
36
47
:8
;9
<10
=11
J12
K13
Q14
R15
S16
T17
e18
f19
l20
m21
n22
o23
x24
y25
Ж
0
1
#2
$3
34
45
:6
;7
J8
K9
Q10
R11
e12
f13
l14
m15
x16
y17
 
Ь
	variables
~metrics
trainable_variables

layers
regularization_losses
Аnon_trainable_variables
 Бlayer_regularization_losses
 
 
 
 
Ю
	variables
Вmetrics
regularization_losses
Гlayers
trainable_variables
Дnon_trainable_variables
 Еlayer_regularization_losses
][
VARIABLE_VALUEsequential_24/conv2d_36/kernel)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_24/conv2d_36/bias'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Ю
	variables
Жmetrics
regularization_losses
Зlayers
 trainable_variables
Иnon_trainable_variables
 Йlayer_regularization_losses
 
hf
VARIABLE_VALUE*sequential_24/batch_normalization_94/gamma(layer-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)sequential_24/batch_normalization_94/beta'layer-2/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE0sequential_24/batch_normalization_94/moving_mean.layer-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE4sequential_24/batch_normalization_94/moving_variance2layer-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
%2
&3
 

#0
$1
Ю
'	variables
Кmetrics
(regularization_losses
Лlayers
)trainable_variables
Мnon_trainable_variables
 Нlayer_regularization_losses
 
 
 
Ю
+	variables
Оmetrics
,regularization_losses
Пlayers
-trainable_variables
Рnon_trainable_variables
 Сlayer_regularization_losses
 
 
 
Ю
/	variables
Тmetrics
0regularization_losses
Уlayers
1trainable_variables
Фnon_trainable_variables
 Хlayer_regularization_losses
][
VARIABLE_VALUEsequential_24/conv2d_37/kernel)layer-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_24/conv2d_37/bias'layer-5/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
Ю
5	variables
Цmetrics
6regularization_losses
Чlayers
7trainable_variables
Шnon_trainable_variables
 Щlayer_regularization_losses
 
hf
VARIABLE_VALUE*sequential_24/batch_normalization_95/gamma(layer-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)sequential_24/batch_normalization_95/beta'layer-6/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE0sequential_24/batch_normalization_95/moving_mean.layer-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE4sequential_24/batch_normalization_95/moving_variance2layer-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
<2
=3
 

:0
;1
Ю
>	variables
Ъmetrics
?regularization_losses
Ыlayers
@trainable_variables
Ьnon_trainable_variables
 Эlayer_regularization_losses
 
 
 
Ю
B	variables
Юmetrics
Cregularization_losses
Яlayers
Dtrainable_variables
аnon_trainable_variables
 бlayer_regularization_losses
 
 
 
Ю
F	variables
вmetrics
Gregularization_losses
гlayers
Htrainable_variables
дnon_trainable_variables
 еlayer_regularization_losses
][
VARIABLE_VALUEsequential_24/conv2d_38/kernel)layer-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_24/conv2d_38/bias'layer-9/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
Ю
L	variables
жmetrics
Mregularization_losses
зlayers
Ntrainable_variables
иnon_trainable_variables
 йlayer_regularization_losses
 
ig
VARIABLE_VALUE*sequential_24/batch_normalization_96/gamma)layer-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE)sequential_24/batch_normalization_96/beta(layer-10/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE0sequential_24/batch_normalization_96/moving_mean/layer-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE4sequential_24/batch_normalization_96/moving_variance3layer-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
S2
T3
 

Q0
R1
Ю
U	variables
кmetrics
Vregularization_losses
лlayers
Wtrainable_variables
мnon_trainable_variables
 нlayer_regularization_losses
 
 
 
Ю
Y	variables
оmetrics
Zregularization_losses
пlayers
[trainable_variables
░non_trainable_variables
 ▒layer_regularization_losses
 
 
 
Ю
]	variables
▓metrics
^regularization_losses
│layers
_trainable_variables
┤non_trainable_variables
 ╡layer_regularization_losses
 
 
 
Ю
a	variables
╢metrics
bregularization_losses
╖layers
ctrainable_variables
╕non_trainable_variables
 ╣layer_regularization_losses
][
VARIABLE_VALUEsequential_24/dense_43/kernel*layer-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_24/dense_43/bias(layer-14/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1
 

e0
f1
Ю
g	variables
║metrics
hregularization_losses
╗layers
itrainable_variables
╝non_trainable_variables
 ╜layer_regularization_losses
 
ig
VARIABLE_VALUE*sequential_24/batch_normalization_97/gamma)layer-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE)sequential_24/batch_normalization_97/beta(layer-15/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE0sequential_24/batch_normalization_97/moving_mean/layer-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE4sequential_24/batch_normalization_97/moving_variance3layer-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

l0
m1
n2
o3
 

l0
m1
Ю
p	variables
╛metrics
qregularization_losses
┐layers
rtrainable_variables
└non_trainable_variables
 ┴layer_regularization_losses
 
 
 
Ю
t	variables
┬metrics
uregularization_losses
├layers
vtrainable_variables
─non_trainable_variables
 ┼layer_regularization_losses
][
VARIABLE_VALUEsequential_24/dense_44/kernel*layer-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_24/dense_44/bias(layer-17/bias/.ATTRIBUTES/VARIABLE_VALUE

x0
y1
 

x0
y1
Ю
z	variables
╞metrics
{regularization_losses
╟layers
|trainable_variables
╚non_trainable_variables
 ╔layer_regularization_losses
 
Ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
8
%0
&1
<2
=3
S4
T5
n6
o7
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
%0
&1
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
<0
=1
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
S0
T1
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
 
 
 
 

n0
o1
 
 
 
 
 
 
 
 
 
К
serving_default_input_1Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
х

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential_24/conv2d_36/kernelsequential_24/conv2d_36/bias*sequential_24/batch_normalization_94/gamma)sequential_24/batch_normalization_94/beta0sequential_24/batch_normalization_94/moving_mean4sequential_24/batch_normalization_94/moving_variancesequential_24/conv2d_37/kernelsequential_24/conv2d_37/bias*sequential_24/batch_normalization_95/gamma)sequential_24/batch_normalization_95/beta0sequential_24/batch_normalization_95/moving_mean4sequential_24/batch_normalization_95/moving_variancesequential_24/conv2d_38/kernelsequential_24/conv2d_38/bias*sequential_24/batch_normalization_96/gamma)sequential_24/batch_normalization_96/beta0sequential_24/batch_normalization_96/moving_mean4sequential_24/batch_normalization_96/moving_variancesequential_24/dense_43/kernelsequential_24/dense_43/bias0sequential_24/batch_normalization_97/moving_mean4sequential_24/batch_normalization_97/moving_variance)sequential_24/batch_normalization_97/beta*sequential_24/batch_normalization_97/gammasequential_24/dense_44/kernelsequential_24/dense_44/bias*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*-
f(R&
$__inference_signature_wrapper_555431
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╪
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2sequential_24/conv2d_36/kernel/Read/ReadVariableOp0sequential_24/conv2d_36/bias/Read/ReadVariableOp>sequential_24/batch_normalization_94/gamma/Read/ReadVariableOp=sequential_24/batch_normalization_94/beta/Read/ReadVariableOpDsequential_24/batch_normalization_94/moving_mean/Read/ReadVariableOpHsequential_24/batch_normalization_94/moving_variance/Read/ReadVariableOp2sequential_24/conv2d_37/kernel/Read/ReadVariableOp0sequential_24/conv2d_37/bias/Read/ReadVariableOp>sequential_24/batch_normalization_95/gamma/Read/ReadVariableOp=sequential_24/batch_normalization_95/beta/Read/ReadVariableOpDsequential_24/batch_normalization_95/moving_mean/Read/ReadVariableOpHsequential_24/batch_normalization_95/moving_variance/Read/ReadVariableOp2sequential_24/conv2d_38/kernel/Read/ReadVariableOp0sequential_24/conv2d_38/bias/Read/ReadVariableOp>sequential_24/batch_normalization_96/gamma/Read/ReadVariableOp=sequential_24/batch_normalization_96/beta/Read/ReadVariableOpDsequential_24/batch_normalization_96/moving_mean/Read/ReadVariableOpHsequential_24/batch_normalization_96/moving_variance/Read/ReadVariableOp1sequential_24/dense_43/kernel/Read/ReadVariableOp/sequential_24/dense_43/bias/Read/ReadVariableOp>sequential_24/batch_normalization_97/gamma/Read/ReadVariableOp=sequential_24/batch_normalization_97/beta/Read/ReadVariableOpDsequential_24/batch_normalization_97/moving_mean/Read/ReadVariableOpHsequential_24/batch_normalization_97/moving_variance/Read/ReadVariableOp1sequential_24/dense_44/kernel/Read/ReadVariableOp/sequential_24/dense_44/bias/Read/ReadVariableOpConst*'
Tin 
2*
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
__inference__traced_save_556441
╦

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential_24/conv2d_36/kernelsequential_24/conv2d_36/bias*sequential_24/batch_normalization_94/gamma)sequential_24/batch_normalization_94/beta0sequential_24/batch_normalization_94/moving_mean4sequential_24/batch_normalization_94/moving_variancesequential_24/conv2d_37/kernelsequential_24/conv2d_37/bias*sequential_24/batch_normalization_95/gamma)sequential_24/batch_normalization_95/beta0sequential_24/batch_normalization_95/moving_mean4sequential_24/batch_normalization_95/moving_variancesequential_24/conv2d_38/kernelsequential_24/conv2d_38/bias*sequential_24/batch_normalization_96/gamma)sequential_24/batch_normalization_96/beta0sequential_24/batch_normalization_96/moving_mean4sequential_24/batch_normalization_96/moving_variancesequential_24/dense_43/kernelsequential_24/dense_43/bias*sequential_24/batch_normalization_97/gamma)sequential_24/batch_normalization_97/beta0sequential_24/batch_normalization_97/moving_mean4sequential_24/batch_normalization_97/moving_variancesequential_24/dense_44/kernelsequential_24/dense_44/bias*&
Tin
2*
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
"__inference__traced_restore_556531ЮЩ
╞b
╞
I__inference_sequential_24_layer_call_and_return_conditional_losses_555192
input_1,
(conv2d_36_statefulpartitionedcall_args_1,
(conv2d_36_statefulpartitionedcall_args_29
5batch_normalization_94_statefulpartitionedcall_args_19
5batch_normalization_94_statefulpartitionedcall_args_29
5batch_normalization_94_statefulpartitionedcall_args_39
5batch_normalization_94_statefulpartitionedcall_args_4,
(conv2d_37_statefulpartitionedcall_args_1,
(conv2d_37_statefulpartitionedcall_args_29
5batch_normalization_95_statefulpartitionedcall_args_19
5batch_normalization_95_statefulpartitionedcall_args_29
5batch_normalization_95_statefulpartitionedcall_args_39
5batch_normalization_95_statefulpartitionedcall_args_4,
(conv2d_38_statefulpartitionedcall_args_1,
(conv2d_38_statefulpartitionedcall_args_29
5batch_normalization_96_statefulpartitionedcall_args_19
5batch_normalization_96_statefulpartitionedcall_args_29
5batch_normalization_96_statefulpartitionedcall_args_39
5batch_normalization_96_statefulpartitionedcall_args_4+
'dense_43_statefulpartitionedcall_args_1+
'dense_43_statefulpartitionedcall_args_29
5batch_normalization_97_statefulpartitionedcall_args_19
5batch_normalization_97_statefulpartitionedcall_args_29
5batch_normalization_97_statefulpartitionedcall_args_39
5batch_normalization_97_statefulpartitionedcall_args_4+
'dense_44_statefulpartitionedcall_args_1+
'dense_44_statefulpartitionedcall_args_2
identityИв.batch_normalization_94/StatefulPartitionedCallв.batch_normalization_95/StatefulPartitionedCallв.batch_normalization_96/StatefulPartitionedCallв.batch_normalization_97/StatefulPartitionedCallв!conv2d_36/StatefulPartitionedCallв!conv2d_37/StatefulPartitionedCallв!conv2d_38/StatefulPartitionedCallв dense_43/StatefulPartitionedCallв dense_44/StatefulPartitionedCallв"dropout_36/StatefulPartitionedCallв"dropout_37/StatefulPartitionedCallв"dropout_38/StatefulPartitionedCallЇ
 up_sampling2d_12/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_5542482"
 up_sampling2d_12/PartitionedCallя
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_12/PartitionedCall:output:0(conv2d_36_statefulpartitionedcall_args_1(conv2d_36_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_36_layer_call_and_return_conditional_losses_5542662#
!conv2d_36/StatefulPartitionedCallб
.batch_normalization_94/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:05batch_normalization_94_statefulpartitionedcall_args_15batch_normalization_94_statefulpartitionedcall_args_25batch_normalization_94_statefulpartitionedcall_args_35batch_normalization_94_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_55436820
.batch_normalization_94/StatefulPartitionedCallЮ
leaky_re_lu_91/PartitionedCallPartitionedCall7batch_normalization_94/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_91_layer_call_and_return_conditional_losses_5548902 
leaky_re_lu_91/PartitionedCallЪ
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_91/PartitionedCall:output:0*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_5549182$
"dropout_36/StatefulPartitionedCallё
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0(conv2d_37_statefulpartitionedcall_args_1(conv2d_37_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_37_layer_call_and_return_conditional_losses_5544182#
!conv2d_37/StatefulPartitionedCallб
.batch_normalization_95/StatefulPartitionedCallStatefulPartitionedCall*conv2d_37/StatefulPartitionedCall:output:05batch_normalization_95_statefulpartitionedcall_args_15batch_normalization_95_statefulpartitionedcall_args_25batch_normalization_95_statefulpartitionedcall_args_35batch_normalization_95_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_55452020
.batch_normalization_95/StatefulPartitionedCallЮ
leaky_re_lu_92/PartitionedCallPartitionedCall7batch_normalization_95/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_92_layer_call_and_return_conditional_losses_5549672 
leaky_re_lu_92/PartitionedCall┐
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_92/PartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_5549952$
"dropout_37/StatefulPartitionedCallЄ
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0(conv2d_38_statefulpartitionedcall_args_1(conv2d_38_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_38_layer_call_and_return_conditional_losses_5545702#
!conv2d_38/StatefulPartitionedCallв
.batch_normalization_96/StatefulPartitionedCallStatefulPartitionedCall*conv2d_38/StatefulPartitionedCall:output:05batch_normalization_96_statefulpartitionedcall_args_15batch_normalization_96_statefulpartitionedcall_args_25batch_normalization_96_statefulpartitionedcall_args_35batch_normalization_96_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_55467220
.batch_normalization_96/StatefulPartitionedCallЯ
leaky_re_lu_93/PartitionedCallPartitionedCall7batch_normalization_96/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_93_layer_call_and_return_conditional_losses_5550442 
leaky_re_lu_93/PartitionedCall└
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_93/PartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_5550722$
"dropout_38/StatefulPartitionedCallї
flatten_12/PartitionedCallPartitionedCall+dropout_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:                  *-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_flatten_12_layer_call_and_return_conditional_losses_5551022
flatten_12/PartitionedCall╩
 dense_43/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0'dense_43_statefulpartitionedcall_args_1'dense_43_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_5551202"
 dense_43/StatefulPartitionedCallЖ
.batch_normalization_97/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:05batch_normalization_97_statefulpartitionedcall_args_15batch_normalization_97_statefulpartitionedcall_args_25batch_normalization_97_statefulpartitionedcall_args_35batch_normalization_97_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_55481520
.batch_normalization_97/StatefulPartitionedCallД
leaky_re_lu_94/PartitionedCallPartitionedCall7batch_normalization_97/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_94_layer_call_and_return_conditional_losses_5551602 
leaky_re_lu_94/PartitionedCall╬
 dense_44/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_94/PartitionedCall:output:0'dense_44_statefulpartitionedcall_args_1'dense_44_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_44_layer_call_and_return_conditional_losses_5551792"
 dense_44/StatefulPartitionedCallт
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0/^batch_normalization_94/StatefulPartitionedCall/^batch_normalization_95/StatefulPartitionedCall/^batch_normalization_96/StatefulPartitionedCall/^batch_normalization_97/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::2`
.batch_normalization_94/StatefulPartitionedCall.batch_normalization_94/StatefulPartitionedCall2`
.batch_normalization_95/StatefulPartitionedCall.batch_normalization_95/StatefulPartitionedCall2`
.batch_normalization_96/StatefulPartitionedCall.batch_normalization_96/StatefulPartitionedCall2`
.batch_normalization_97/StatefulPartitionedCall.batch_normalization_97/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
▌]
╓
I__inference_sequential_24_layer_call_and_return_conditional_losses_555370

inputs,
(conv2d_36_statefulpartitionedcall_args_1,
(conv2d_36_statefulpartitionedcall_args_29
5batch_normalization_94_statefulpartitionedcall_args_19
5batch_normalization_94_statefulpartitionedcall_args_29
5batch_normalization_94_statefulpartitionedcall_args_39
5batch_normalization_94_statefulpartitionedcall_args_4,
(conv2d_37_statefulpartitionedcall_args_1,
(conv2d_37_statefulpartitionedcall_args_29
5batch_normalization_95_statefulpartitionedcall_args_19
5batch_normalization_95_statefulpartitionedcall_args_29
5batch_normalization_95_statefulpartitionedcall_args_39
5batch_normalization_95_statefulpartitionedcall_args_4,
(conv2d_38_statefulpartitionedcall_args_1,
(conv2d_38_statefulpartitionedcall_args_29
5batch_normalization_96_statefulpartitionedcall_args_19
5batch_normalization_96_statefulpartitionedcall_args_29
5batch_normalization_96_statefulpartitionedcall_args_39
5batch_normalization_96_statefulpartitionedcall_args_4+
'dense_43_statefulpartitionedcall_args_1+
'dense_43_statefulpartitionedcall_args_29
5batch_normalization_97_statefulpartitionedcall_args_19
5batch_normalization_97_statefulpartitionedcall_args_29
5batch_normalization_97_statefulpartitionedcall_args_39
5batch_normalization_97_statefulpartitionedcall_args_4+
'dense_44_statefulpartitionedcall_args_1+
'dense_44_statefulpartitionedcall_args_2
identityИв.batch_normalization_94/StatefulPartitionedCallв.batch_normalization_95/StatefulPartitionedCallв.batch_normalization_96/StatefulPartitionedCallв.batch_normalization_97/StatefulPartitionedCallв!conv2d_36/StatefulPartitionedCallв!conv2d_37/StatefulPartitionedCallв!conv2d_38/StatefulPartitionedCallв dense_43/StatefulPartitionedCallв dense_44/StatefulPartitionedCallє
 up_sampling2d_12/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_5542482"
 up_sampling2d_12/PartitionedCallя
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_12/PartitionedCall:output:0(conv2d_36_statefulpartitionedcall_args_1(conv2d_36_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_36_layer_call_and_return_conditional_losses_5542662#
!conv2d_36/StatefulPartitionedCallб
.batch_normalization_94/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:05batch_normalization_94_statefulpartitionedcall_args_15batch_normalization_94_statefulpartitionedcall_args_25batch_normalization_94_statefulpartitionedcall_args_35batch_normalization_94_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_55439920
.batch_normalization_94/StatefulPartitionedCallЮ
leaky_re_lu_91/PartitionedCallPartitionedCall7batch_normalization_94/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_91_layer_call_and_return_conditional_losses_5548902 
leaky_re_lu_91/PartitionedCallВ
dropout_36/PartitionedCallPartitionedCall'leaky_re_lu_91/PartitionedCall:output:0*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_5549232
dropout_36/PartitionedCallщ
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0(conv2d_37_statefulpartitionedcall_args_1(conv2d_37_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_37_layer_call_and_return_conditional_losses_5544182#
!conv2d_37/StatefulPartitionedCallб
.batch_normalization_95/StatefulPartitionedCallStatefulPartitionedCall*conv2d_37/StatefulPartitionedCall:output:05batch_normalization_95_statefulpartitionedcall_args_15batch_normalization_95_statefulpartitionedcall_args_25batch_normalization_95_statefulpartitionedcall_args_35batch_normalization_95_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_55455120
.batch_normalization_95/StatefulPartitionedCallЮ
leaky_re_lu_92/PartitionedCallPartitionedCall7batch_normalization_95/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_92_layer_call_and_return_conditional_losses_5549672 
leaky_re_lu_92/PartitionedCallВ
dropout_37/PartitionedCallPartitionedCall'leaky_re_lu_92/PartitionedCall:output:0*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_5550002
dropout_37/PartitionedCallъ
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0(conv2d_38_statefulpartitionedcall_args_1(conv2d_38_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_38_layer_call_and_return_conditional_losses_5545702#
!conv2d_38/StatefulPartitionedCallв
.batch_normalization_96/StatefulPartitionedCallStatefulPartitionedCall*conv2d_38/StatefulPartitionedCall:output:05batch_normalization_96_statefulpartitionedcall_args_15batch_normalization_96_statefulpartitionedcall_args_25batch_normalization_96_statefulpartitionedcall_args_35batch_normalization_96_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_55470320
.batch_normalization_96/StatefulPartitionedCallЯ
leaky_re_lu_93/PartitionedCallPartitionedCall7batch_normalization_96/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_93_layer_call_and_return_conditional_losses_5550442 
leaky_re_lu_93/PartitionedCallГ
dropout_38/PartitionedCallPartitionedCall'leaky_re_lu_93/PartitionedCall:output:0*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_5550772
dropout_38/PartitionedCallэ
flatten_12/PartitionedCallPartitionedCall#dropout_38/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:                  *-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_flatten_12_layer_call_and_return_conditional_losses_5551022
flatten_12/PartitionedCall╩
 dense_43/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0'dense_43_statefulpartitionedcall_args_1'dense_43_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_5551202"
 dense_43/StatefulPartitionedCallЖ
.batch_normalization_97/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:05batch_normalization_97_statefulpartitionedcall_args_15batch_normalization_97_statefulpartitionedcall_args_25batch_normalization_97_statefulpartitionedcall_args_35batch_normalization_97_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_55484720
.batch_normalization_97/StatefulPartitionedCallД
leaky_re_lu_94/PartitionedCallPartitionedCall7batch_normalization_97/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_94_layer_call_and_return_conditional_losses_5551602 
leaky_re_lu_94/PartitionedCall╬
 dense_44/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_94/PartitionedCall:output:0'dense_44_statefulpartitionedcall_args_1'dense_44_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_44_layer_call_and_return_conditional_losses_5551792"
 dense_44/StatefulPartitionedCallє
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0/^batch_normalization_94/StatefulPartitionedCall/^batch_normalization_95/StatefulPartitionedCall/^batch_normalization_96/StatefulPartitionedCall/^batch_normalization_97/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::2`
.batch_normalization_94/StatefulPartitionedCall.batch_normalization_94/StatefulPartitionedCall2`
.batch_normalization_95/StatefulPartitionedCall.batch_normalization_95/StatefulPartitionedCall2`
.batch_normalization_96/StatefulPartitionedCall.batch_normalization_96/StatefulPartitionedCall2`
.batch_normalization_97/StatefulPartitionedCall.batch_normalization_97/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
щ
f
J__inference_leaky_re_lu_92_layer_call_and_return_conditional_losses_554967

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
╗$
Э
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_554368

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_554353
assignmovingavg_1_554360
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
loc:@AssignMovingAvg/554353*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xп
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/554353*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_554353*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp╠
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/554353*
_output_shapes
:@2
AssignMovingAvg/sub_1╡
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/554353*
_output_shapes
:@2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_554353AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/554353*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/554360*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╖
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/554360*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_554360*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╪
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/554360*
_output_shapes
:@2
AssignMovingAvg_1/sub_1┐
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/554360*
_output_shapes
:@2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_554360AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/554360*
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
є
▌
D__inference_dense_43_layer_call_and_return_conditional_losses_555120

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А1*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:                  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
▒
K
/__inference_leaky_re_lu_92_layer_call_fn_556043

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
J__inference_leaky_re_lu_92_layer_call_and_return_conditional_losses_5549672
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
°

▐
E__inference_conv2d_38_layer_call_and_return_conditional_losses_554570

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd░
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Д
а
.__inference_sequential_24_layer_call_fn_555809

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
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identityИвStatefulPartitionedCall╖	
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_5552912
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ф

b
F__inference_flatten_12_layer_call_and_return_conditional_losses_555102

inputs
identityD
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
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         2
Reshape/shape/1Ж
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:                  2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:& "
 
_user_specified_nameinputs
┼
л
*__inference_conv2d_36_layer_call_fn_554274

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_36_layer_call_and_return_conditional_losses_5542662
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ї
А
7__inference_batch_normalization_96_layer_call_fn_556143

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
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_5546722
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
╩$
Э
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_556112

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_556097
assignmovingavg_1_556104
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
loc:@AssignMovingAvg/556097*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xп
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/556097*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_556097*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/556097*
_output_shapes	
:А2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/556097*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_556097AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/556097*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/556104*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╖
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/556104*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_556104*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/556104*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/556104*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_556104AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/556104*
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
■
к
)__inference_dense_43_layer_call_fn_556231

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_5551202
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:                  ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ф

b
F__inference_flatten_12_layer_call_and_return_conditional_losses_556209

inputs
identityD
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
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         2
Reshape/shape/1Ж
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:                  2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:& "
 
_user_specified_nameinputs
ё
А
7__inference_batch_normalization_95_layer_call_fn_556033

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
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_5545512
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
ч>
╓
__inference__traced_save_556441
file_prefix=
9savev2_sequential_24_conv2d_36_kernel_read_readvariableop;
7savev2_sequential_24_conv2d_36_bias_read_readvariableopI
Esavev2_sequential_24_batch_normalization_94_gamma_read_readvariableopH
Dsavev2_sequential_24_batch_normalization_94_beta_read_readvariableopO
Ksavev2_sequential_24_batch_normalization_94_moving_mean_read_readvariableopS
Osavev2_sequential_24_batch_normalization_94_moving_variance_read_readvariableop=
9savev2_sequential_24_conv2d_37_kernel_read_readvariableop;
7savev2_sequential_24_conv2d_37_bias_read_readvariableopI
Esavev2_sequential_24_batch_normalization_95_gamma_read_readvariableopH
Dsavev2_sequential_24_batch_normalization_95_beta_read_readvariableopO
Ksavev2_sequential_24_batch_normalization_95_moving_mean_read_readvariableopS
Osavev2_sequential_24_batch_normalization_95_moving_variance_read_readvariableop=
9savev2_sequential_24_conv2d_38_kernel_read_readvariableop;
7savev2_sequential_24_conv2d_38_bias_read_readvariableopI
Esavev2_sequential_24_batch_normalization_96_gamma_read_readvariableopH
Dsavev2_sequential_24_batch_normalization_96_beta_read_readvariableopO
Ksavev2_sequential_24_batch_normalization_96_moving_mean_read_readvariableopS
Osavev2_sequential_24_batch_normalization_96_moving_variance_read_readvariableop<
8savev2_sequential_24_dense_43_kernel_read_readvariableop:
6savev2_sequential_24_dense_43_bias_read_readvariableopI
Esavev2_sequential_24_batch_normalization_97_gamma_read_readvariableopH
Dsavev2_sequential_24_batch_normalization_97_beta_read_readvariableopO
Ksavev2_sequential_24_batch_normalization_97_moving_mean_read_readvariableopS
Osavev2_sequential_24_batch_normalization_97_moving_variance_read_readvariableop<
8savev2_sequential_24_dense_44_kernel_read_readvariableop:
6savev2_sequential_24_dense_44_bias_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1е
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_0e49d141f6404dfaa8ecc1e2949cb1db/part2
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
ShardedFilenameП

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*б	
valueЧ	BФ	B)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB'layer-2/beta/.ATTRIBUTES/VARIABLE_VALUEB.layer-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2layer-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)layer-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-5/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB'layer-6/beta/.ATTRIBUTES/VARIABLE_VALUEB.layer-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2layer-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)layer-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB(layer-10/beta/.ATTRIBUTES/VARIABLE_VALUEB/layer-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3layer-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB*layer-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB(layer-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB(layer-15/beta/.ATTRIBUTES/VARIABLE_VALUEB/layer-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3layer-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB*layer-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB(layer-17/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names╝
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices▒
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_sequential_24_conv2d_36_kernel_read_readvariableop7savev2_sequential_24_conv2d_36_bias_read_readvariableopEsavev2_sequential_24_batch_normalization_94_gamma_read_readvariableopDsavev2_sequential_24_batch_normalization_94_beta_read_readvariableopKsavev2_sequential_24_batch_normalization_94_moving_mean_read_readvariableopOsavev2_sequential_24_batch_normalization_94_moving_variance_read_readvariableop9savev2_sequential_24_conv2d_37_kernel_read_readvariableop7savev2_sequential_24_conv2d_37_bias_read_readvariableopEsavev2_sequential_24_batch_normalization_95_gamma_read_readvariableopDsavev2_sequential_24_batch_normalization_95_beta_read_readvariableopKsavev2_sequential_24_batch_normalization_95_moving_mean_read_readvariableopOsavev2_sequential_24_batch_normalization_95_moving_variance_read_readvariableop9savev2_sequential_24_conv2d_38_kernel_read_readvariableop7savev2_sequential_24_conv2d_38_bias_read_readvariableopEsavev2_sequential_24_batch_normalization_96_gamma_read_readvariableopDsavev2_sequential_24_batch_normalization_96_beta_read_readvariableopKsavev2_sequential_24_batch_normalization_96_moving_mean_read_readvariableopOsavev2_sequential_24_batch_normalization_96_moving_variance_read_readvariableop8savev2_sequential_24_dense_43_kernel_read_readvariableop6savev2_sequential_24_dense_43_bias_read_readvariableopEsavev2_sequential_24_batch_normalization_97_gamma_read_readvariableopDsavev2_sequential_24_batch_normalization_97_beta_read_readvariableopKsavev2_sequential_24_batch_normalization_97_moving_mean_read_readvariableopOsavev2_sequential_24_batch_normalization_97_moving_variance_read_readvariableop8savev2_sequential_24_dense_44_kernel_read_readvariableop6savev2_sequential_24_dense_44_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *(
dtypes
22
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

identity_1Identity_1:output:0*ш
_input_shapes╓
╙: :@:@:@:@:@:@:@@:@:@:@:@:@:@А:А:А:А:А:А:	А1:::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
В
ї
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_556015

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
є
▌
D__inference_dense_43_layer_call_and_return_conditional_losses_556224

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А1*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:                  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
п.
▒
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_556270

inputs
assignmovingavg_556245
assignmovingavg_1_556251 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвCast/ReadVariableOpвCast_1/ReadVariableOp^
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
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/556245*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_556245*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp├
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/556245*
_output_shapes
:2
AssignMovingAvg/sub║
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/556245*
_output_shapes
:2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_556245AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/556245*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/556251*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_556251*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp═
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/556251*
_output_shapes
:2
AssignMovingAvg_1/sub─
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/556251*
_output_shapes
:2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_556251AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/556251*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1з
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:& "
 
_user_specified_nameinputs
С
▌
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_554847

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityИвCast/ReadVariableOpвCast_1/ReadVariableOpвCast_2/ReadVariableOpвCast_3/ReadVariableOp^
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

LogicalAndГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЕ
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1┼
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:& "
 
_user_specified_nameinputs
Ъ
f
J__inference_leaky_re_lu_94_layer_call_and_return_conditional_losses_555160

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
В
ї
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_555896

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
╗$
Э
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_555874

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_555859
assignmovingavg_1_555866
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
loc:@AssignMovingAvg/555859*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xп
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/555859*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_555859*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp╠
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/555859*
_output_shapes
:@2
AssignMovingAvg/sub_1╡
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/555859*
_output_shapes
:@2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_555859AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/555859*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/555866*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╖
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/555866*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_555866*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╪
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/555866*
_output_shapes
:@2
AssignMovingAvg_1/sub_1┐
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/555866*
_output_shapes
:@2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_555866AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/555866*
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
З
б
.__inference_sequential_24_layer_call_fn_555320
input_1"
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
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identityИвStatefulPartitionedCall╕	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_5552912
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
тк
т
I__inference_sequential_24_layer_call_and_return_conditional_losses_555778

inputs,
(conv2d_36_conv2d_readvariableop_resource-
)conv2d_36_biasadd_readvariableop_resource2
.batch_normalization_94_readvariableop_resource4
0batch_normalization_94_readvariableop_1_resourceC
?batch_normalization_94_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_94_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_37_conv2d_readvariableop_resource-
)conv2d_37_biasadd_readvariableop_resource2
.batch_normalization_95_readvariableop_resource4
0batch_normalization_95_readvariableop_1_resourceC
?batch_normalization_95_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_95_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_38_conv2d_readvariableop_resource-
)conv2d_38_biasadd_readvariableop_resource2
.batch_normalization_96_readvariableop_resource4
0batch_normalization_96_readvariableop_1_resourceC
?batch_normalization_96_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_96_fusedbatchnormv3_readvariableop_1_resource+
'dense_43_matmul_readvariableop_resource,
(dense_43_biasadd_readvariableop_resource7
3batch_normalization_97_cast_readvariableop_resource9
5batch_normalization_97_cast_1_readvariableop_resource9
5batch_normalization_97_cast_2_readvariableop_resource9
5batch_normalization_97_cast_3_readvariableop_resource+
'dense_44_matmul_readvariableop_resource,
(dense_44_biasadd_readvariableop_resource
identityИв6batch_normalization_94/FusedBatchNormV3/ReadVariableOpв8batch_normalization_94/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_94/ReadVariableOpв'batch_normalization_94/ReadVariableOp_1в6batch_normalization_95/FusedBatchNormV3/ReadVariableOpв8batch_normalization_95/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_95/ReadVariableOpв'batch_normalization_95/ReadVariableOp_1в6batch_normalization_96/FusedBatchNormV3/ReadVariableOpв8batch_normalization_96/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_96/ReadVariableOpв'batch_normalization_96/ReadVariableOp_1в*batch_normalization_97/Cast/ReadVariableOpв,batch_normalization_97/Cast_1/ReadVariableOpв,batch_normalization_97/Cast_2/ReadVariableOpв,batch_normalization_97/Cast_3/ReadVariableOpв conv2d_36/BiasAdd/ReadVariableOpвconv2d_36/Conv2D/ReadVariableOpв conv2d_37/BiasAdd/ReadVariableOpвconv2d_37/Conv2D/ReadVariableOpв conv2d_38/BiasAdd/ReadVariableOpвconv2d_38/Conv2D/ReadVariableOpвdense_43/BiasAdd/ReadVariableOpвdense_43/MatMul/ReadVariableOpвdense_44/BiasAdd/ReadVariableOpвdense_44/MatMul/ReadVariableOpf
up_sampling2d_12/ShapeShapeinputs*
T0*
_output_shapes
:2
up_sampling2d_12/ShapeЦ
$up_sampling2d_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$up_sampling2d_12/strided_slice/stackЪ
&up_sampling2d_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_12/strided_slice/stack_1Ъ
&up_sampling2d_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_12/strided_slice/stack_2┤
up_sampling2d_12/strided_sliceStridedSliceup_sampling2d_12/Shape:output:0-up_sampling2d_12/strided_slice/stack:output:0/up_sampling2d_12/strided_slice/stack_1:output:0/up_sampling2d_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
up_sampling2d_12/strided_sliceБ
up_sampling2d_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_12/Constв
up_sampling2d_12/mulMul'up_sampling2d_12/strided_slice:output:0up_sampling2d_12/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_12/mulэ
-up_sampling2d_12/resize/ResizeNearestNeighborResizeNearestNeighborinputsup_sampling2d_12/mul:z:0*
T0*/
_output_shapes
:         88*
half_pixel_centers(2/
-up_sampling2d_12/resize/ResizeNearestNeighbor│
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_36/Conv2D/ReadVariableOp∙
conv2d_36/Conv2DConv2D>up_sampling2d_12/resize/ResizeNearestNeighbor:resized_images:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_36/Conv2Dк
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp░
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_36/BiasAddМ
#batch_normalization_94/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_94/LogicalAnd/xМ
#batch_normalization_94/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_94/LogicalAnd/y╚
!batch_normalization_94/LogicalAnd
LogicalAnd,batch_normalization_94/LogicalAnd/x:output:0,batch_normalization_94/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_94/LogicalAnd╣
%batch_normalization_94/ReadVariableOpReadVariableOp.batch_normalization_94_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_94/ReadVariableOp┐
'batch_normalization_94/ReadVariableOp_1ReadVariableOp0batch_normalization_94_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_94/ReadVariableOp_1ь
6batch_normalization_94/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_94_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_94/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_94/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_94_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_94/FusedBatchNormV3/ReadVariableOp_1ш
'batch_normalization_94/FusedBatchNormV3FusedBatchNormV3conv2d_36/BiasAdd:output:0-batch_normalization_94/ReadVariableOp:value:0/batch_normalization_94/ReadVariableOp_1:value:0>batch_normalization_94/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_94/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_94/FusedBatchNormV3Б
batch_normalization_94/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
batch_normalization_94/ConstЯ
leaky_re_lu_91/LeakyRelu	LeakyRelu+batch_normalization_94/FusedBatchNormV3:y:0*/
_output_shapes
:         @2
leaky_re_lu_91/LeakyReluШ
dropout_36/IdentityIdentity&leaky_re_lu_91/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @2
dropout_36/Identity│
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_37/Conv2D/ReadVariableOp╫
conv2d_37/Conv2DConv2Ddropout_36/Identity:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_37/Conv2Dк
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp░
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_37/BiasAddМ
#batch_normalization_95/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_95/LogicalAnd/xМ
#batch_normalization_95/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_95/LogicalAnd/y╚
!batch_normalization_95/LogicalAnd
LogicalAnd,batch_normalization_95/LogicalAnd/x:output:0,batch_normalization_95/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_95/LogicalAnd╣
%batch_normalization_95/ReadVariableOpReadVariableOp.batch_normalization_95_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_95/ReadVariableOp┐
'batch_normalization_95/ReadVariableOp_1ReadVariableOp0batch_normalization_95_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_95/ReadVariableOp_1ь
6batch_normalization_95/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_95_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_95/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_95/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_95_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_95/FusedBatchNormV3/ReadVariableOp_1ш
'batch_normalization_95/FusedBatchNormV3FusedBatchNormV3conv2d_37/BiasAdd:output:0-batch_normalization_95/ReadVariableOp:value:0/batch_normalization_95/ReadVariableOp_1:value:0>batch_normalization_95/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_95/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_95/FusedBatchNormV3Б
batch_normalization_95/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
batch_normalization_95/ConstЯ
leaky_re_lu_92/LeakyRelu	LeakyRelu+batch_normalization_95/FusedBatchNormV3:y:0*/
_output_shapes
:         @2
leaky_re_lu_92/LeakyReluШ
dropout_37/IdentityIdentity&leaky_re_lu_92/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @2
dropout_37/Identity┤
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02!
conv2d_38/Conv2D/ReadVariableOp╪
conv2d_38/Conv2DConv2Ddropout_37/Identity:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_38/Conv2Dл
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp▒
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_38/BiasAddМ
#batch_normalization_96/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_96/LogicalAnd/xМ
#batch_normalization_96/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_96/LogicalAnd/y╚
!batch_normalization_96/LogicalAnd
LogicalAnd,batch_normalization_96/LogicalAnd/x:output:0,batch_normalization_96/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_96/LogicalAnd║
%batch_normalization_96/ReadVariableOpReadVariableOp.batch_normalization_96_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_96/ReadVariableOp└
'batch_normalization_96/ReadVariableOp_1ReadVariableOp0batch_normalization_96_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_96/ReadVariableOp_1э
6batch_normalization_96/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_96_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_96/FusedBatchNormV3/ReadVariableOpє
8batch_normalization_96/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_96_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8batch_normalization_96/FusedBatchNormV3/ReadVariableOp_1э
'batch_normalization_96/FusedBatchNormV3FusedBatchNormV3conv2d_38/BiasAdd:output:0-batch_normalization_96/ReadVariableOp:value:0/batch_normalization_96/ReadVariableOp_1:value:0>batch_normalization_96/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_96/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_96/FusedBatchNormV3Б
batch_normalization_96/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
batch_normalization_96/Constа
leaky_re_lu_93/LeakyRelu	LeakyRelu+batch_normalization_96/FusedBatchNormV3:y:0*0
_output_shapes
:         А2
leaky_re_lu_93/LeakyReluЩ
dropout_38/IdentityIdentity&leaky_re_lu_93/LeakyRelu:activations:0*
T0*0
_output_shapes
:         А2
dropout_38/Identityu
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  2
flatten_12/ConstЯ
flatten_12/ReshapeReshapedropout_38/Identity:output:0flatten_12/Const:output:0*
T0*(
_output_shapes
:         А12
flatten_12/Reshapeй
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes
:	А1*
dtype02 
dense_43/MatMul/ReadVariableOpг
dense_43/MatMulMatMulflatten_12/Reshape:output:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_43/MatMulз
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_43/BiasAdd/ReadVariableOpе
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_43/BiasAddМ
#batch_normalization_97/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_97/LogicalAnd/xМ
#batch_normalization_97/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_97/LogicalAnd/y╚
!batch_normalization_97/LogicalAnd
LogicalAnd,batch_normalization_97/LogicalAnd/x:output:0,batch_normalization_97/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_97/LogicalAnd╚
*batch_normalization_97/Cast/ReadVariableOpReadVariableOp3batch_normalization_97_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_97/Cast/ReadVariableOp╬
,batch_normalization_97/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_97_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_97/Cast_1/ReadVariableOp╬
,batch_normalization_97/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_97_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_97/Cast_2/ReadVariableOp╬
,batch_normalization_97/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_97_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_97/Cast_3/ReadVariableOpХ
&batch_normalization_97/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2(
&batch_normalization_97/batchnorm/add/yс
$batch_normalization_97/batchnorm/addAddV24batch_normalization_97/Cast_1/ReadVariableOp:value:0/batch_normalization_97/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_97/batchnorm/addи
&batch_normalization_97/batchnorm/RsqrtRsqrt(batch_normalization_97/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_97/batchnorm/Rsqrt┌
$batch_normalization_97/batchnorm/mulMul*batch_normalization_97/batchnorm/Rsqrt:y:04batch_normalization_97/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_97/batchnorm/mul╬
&batch_normalization_97/batchnorm/mul_1Muldense_43/BiasAdd:output:0(batch_normalization_97/batchnorm/mul:z:0*
T0*'
_output_shapes
:         2(
&batch_normalization_97/batchnorm/mul_1┌
&batch_normalization_97/batchnorm/mul_2Mul2batch_normalization_97/Cast/ReadVariableOp:value:0(batch_normalization_97/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_97/batchnorm/mul_2┌
$batch_normalization_97/batchnorm/subSub4batch_normalization_97/Cast_2/ReadVariableOp:value:0*batch_normalization_97/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_97/batchnorm/subс
&batch_normalization_97/batchnorm/add_1AddV2*batch_normalization_97/batchnorm/mul_1:z:0(batch_normalization_97/batchnorm/sub:z:0*
T0*'
_output_shapes
:         2(
&batch_normalization_97/batchnorm/add_1Ц
leaky_re_lu_94/LeakyRelu	LeakyRelu*batch_normalization_97/batchnorm/add_1:z:0*'
_output_shapes
:         2
leaky_re_lu_94/LeakyReluи
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_44/MatMul/ReadVariableOpо
dense_44/MatMulMatMul&leaky_re_lu_94/LeakyRelu:activations:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/MatMulз
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOpе
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/BiasAdd|
dense_44/SigmoidSigmoiddense_44/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_44/Sigmoid╔	
IdentityIdentitydense_44/Sigmoid:y:07^batch_normalization_94/FusedBatchNormV3/ReadVariableOp9^batch_normalization_94/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_94/ReadVariableOp(^batch_normalization_94/ReadVariableOp_17^batch_normalization_95/FusedBatchNormV3/ReadVariableOp9^batch_normalization_95/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_95/ReadVariableOp(^batch_normalization_95/ReadVariableOp_17^batch_normalization_96/FusedBatchNormV3/ReadVariableOp9^batch_normalization_96/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_96/ReadVariableOp(^batch_normalization_96/ReadVariableOp_1+^batch_normalization_97/Cast/ReadVariableOp-^batch_normalization_97/Cast_1/ReadVariableOp-^batch_normalization_97/Cast_2/ReadVariableOp-^batch_normalization_97/Cast_3/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::2p
6batch_normalization_94/FusedBatchNormV3/ReadVariableOp6batch_normalization_94/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_94/FusedBatchNormV3/ReadVariableOp_18batch_normalization_94/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_94/ReadVariableOp%batch_normalization_94/ReadVariableOp2R
'batch_normalization_94/ReadVariableOp_1'batch_normalization_94/ReadVariableOp_12p
6batch_normalization_95/FusedBatchNormV3/ReadVariableOp6batch_normalization_95/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_95/FusedBatchNormV3/ReadVariableOp_18batch_normalization_95/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_95/ReadVariableOp%batch_normalization_95/ReadVariableOp2R
'batch_normalization_95/ReadVariableOp_1'batch_normalization_95/ReadVariableOp_12p
6batch_normalization_96/FusedBatchNormV3/ReadVariableOp6batch_normalization_96/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_96/FusedBatchNormV3/ReadVariableOp_18batch_normalization_96/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_96/ReadVariableOp%batch_normalization_96/ReadVariableOp2R
'batch_normalization_96/ReadVariableOp_1'batch_normalization_96/ReadVariableOp_12X
*batch_normalization_97/Cast/ReadVariableOp*batch_normalization_97/Cast/ReadVariableOp2\
,batch_normalization_97/Cast_1/ReadVariableOp,batch_normalization_97/Cast_1/ReadVariableOp2\
,batch_normalization_97/Cast_2/ReadVariableOp,batch_normalization_97/Cast_2/ReadVariableOp2\
,batch_normalization_97/Cast_3/ReadVariableOp,batch_normalization_97/Cast_3/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Є
d
F__inference_dropout_38_layer_call_and_return_conditional_losses_556187

inputs

identity_1u
IdentityIdentityinputs*
T0*B
_output_shapes0
.:,                           А2

IdentityД

Identity_1IdentityIdentity:output:0*
T0*B
_output_shapes0
.:,                           А2

Identity_1"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,                           А:& "
 
_user_specified_nameinputs
эц
ц
I__inference_sequential_24_layer_call_and_return_conditional_losses_555653

inputs,
(conv2d_36_conv2d_readvariableop_resource-
)conv2d_36_biasadd_readvariableop_resource2
.batch_normalization_94_readvariableop_resource4
0batch_normalization_94_readvariableop_1_resource1
-batch_normalization_94_assignmovingavg_5554643
/batch_normalization_94_assignmovingavg_1_555471,
(conv2d_37_conv2d_readvariableop_resource-
)conv2d_37_biasadd_readvariableop_resource2
.batch_normalization_95_readvariableop_resource4
0batch_normalization_95_readvariableop_1_resource1
-batch_normalization_95_assignmovingavg_5555173
/batch_normalization_95_assignmovingavg_1_555524,
(conv2d_38_conv2d_readvariableop_resource-
)conv2d_38_biasadd_readvariableop_resource2
.batch_normalization_96_readvariableop_resource4
0batch_normalization_96_readvariableop_1_resource1
-batch_normalization_96_assignmovingavg_5555703
/batch_normalization_96_assignmovingavg_1_555577+
'dense_43_matmul_readvariableop_resource,
(dense_43_biasadd_readvariableop_resource1
-batch_normalization_97_assignmovingavg_5556203
/batch_normalization_97_assignmovingavg_1_5556267
3batch_normalization_97_cast_readvariableop_resource9
5batch_normalization_97_cast_1_readvariableop_resource+
'dense_44_matmul_readvariableop_resource,
(dense_44_biasadd_readvariableop_resource
identityИв:batch_normalization_94/AssignMovingAvg/AssignSubVariableOpв5batch_normalization_94/AssignMovingAvg/ReadVariableOpв<batch_normalization_94/AssignMovingAvg_1/AssignSubVariableOpв7batch_normalization_94/AssignMovingAvg_1/ReadVariableOpв%batch_normalization_94/ReadVariableOpв'batch_normalization_94/ReadVariableOp_1в:batch_normalization_95/AssignMovingAvg/AssignSubVariableOpв5batch_normalization_95/AssignMovingAvg/ReadVariableOpв<batch_normalization_95/AssignMovingAvg_1/AssignSubVariableOpв7batch_normalization_95/AssignMovingAvg_1/ReadVariableOpв%batch_normalization_95/ReadVariableOpв'batch_normalization_95/ReadVariableOp_1в:batch_normalization_96/AssignMovingAvg/AssignSubVariableOpв5batch_normalization_96/AssignMovingAvg/ReadVariableOpв<batch_normalization_96/AssignMovingAvg_1/AssignSubVariableOpв7batch_normalization_96/AssignMovingAvg_1/ReadVariableOpв%batch_normalization_96/ReadVariableOpв'batch_normalization_96/ReadVariableOp_1в:batch_normalization_97/AssignMovingAvg/AssignSubVariableOpв5batch_normalization_97/AssignMovingAvg/ReadVariableOpв<batch_normalization_97/AssignMovingAvg_1/AssignSubVariableOpв7batch_normalization_97/AssignMovingAvg_1/ReadVariableOpв*batch_normalization_97/Cast/ReadVariableOpв,batch_normalization_97/Cast_1/ReadVariableOpв conv2d_36/BiasAdd/ReadVariableOpвconv2d_36/Conv2D/ReadVariableOpв conv2d_37/BiasAdd/ReadVariableOpвconv2d_37/Conv2D/ReadVariableOpв conv2d_38/BiasAdd/ReadVariableOpвconv2d_38/Conv2D/ReadVariableOpвdense_43/BiasAdd/ReadVariableOpвdense_43/MatMul/ReadVariableOpвdense_44/BiasAdd/ReadVariableOpвdense_44/MatMul/ReadVariableOpf
up_sampling2d_12/ShapeShapeinputs*
T0*
_output_shapes
:2
up_sampling2d_12/ShapeЦ
$up_sampling2d_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$up_sampling2d_12/strided_slice/stackЪ
&up_sampling2d_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_12/strided_slice/stack_1Ъ
&up_sampling2d_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_12/strided_slice/stack_2┤
up_sampling2d_12/strided_sliceStridedSliceup_sampling2d_12/Shape:output:0-up_sampling2d_12/strided_slice/stack:output:0/up_sampling2d_12/strided_slice/stack_1:output:0/up_sampling2d_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
up_sampling2d_12/strided_sliceБ
up_sampling2d_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_12/Constв
up_sampling2d_12/mulMul'up_sampling2d_12/strided_slice:output:0up_sampling2d_12/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_12/mulэ
-up_sampling2d_12/resize/ResizeNearestNeighborResizeNearestNeighborinputsup_sampling2d_12/mul:z:0*
T0*/
_output_shapes
:         88*
half_pixel_centers(2/
-up_sampling2d_12/resize/ResizeNearestNeighbor│
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_36/Conv2D/ReadVariableOp∙
conv2d_36/Conv2DConv2D>up_sampling2d_12/resize/ResizeNearestNeighbor:resized_images:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_36/Conv2Dк
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp░
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_36/BiasAddМ
#batch_normalization_94/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_94/LogicalAnd/xМ
#batch_normalization_94/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_94/LogicalAnd/y╚
!batch_normalization_94/LogicalAnd
LogicalAnd,batch_normalization_94/LogicalAnd/x:output:0,batch_normalization_94/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_94/LogicalAnd╣
%batch_normalization_94/ReadVariableOpReadVariableOp.batch_normalization_94_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_94/ReadVariableOp┐
'batch_normalization_94/ReadVariableOp_1ReadVariableOp0batch_normalization_94_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_94/ReadVariableOp_1
batch_normalization_94/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_94/ConstГ
batch_normalization_94/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_94/Const_1г
'batch_normalization_94/FusedBatchNormV3FusedBatchNormV3conv2d_36/BiasAdd:output:0-batch_normalization_94/ReadVariableOp:value:0/batch_normalization_94/ReadVariableOp_1:value:0%batch_normalization_94/Const:output:0'batch_normalization_94/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:2)
'batch_normalization_94/FusedBatchNormV3Е
batch_normalization_94/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2 
batch_normalization_94/Const_2у
,batch_normalization_94/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_94/AssignMovingAvg/555464*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,batch_normalization_94/AssignMovingAvg/sub/xв
*batch_normalization_94/AssignMovingAvg/subSub5batch_normalization_94/AssignMovingAvg/sub/x:output:0'batch_normalization_94/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_94/AssignMovingAvg/555464*
_output_shapes
: 2,
*batch_normalization_94/AssignMovingAvg/sub╪
5batch_normalization_94/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_94_assignmovingavg_555464*
_output_shapes
:@*
dtype027
5batch_normalization_94/AssignMovingAvg/ReadVariableOp┐
,batch_normalization_94/AssignMovingAvg/sub_1Sub=batch_normalization_94/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_94/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_94/AssignMovingAvg/555464*
_output_shapes
:@2.
,batch_normalization_94/AssignMovingAvg/sub_1и
*batch_normalization_94/AssignMovingAvg/mulMul0batch_normalization_94/AssignMovingAvg/sub_1:z:0.batch_normalization_94/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_94/AssignMovingAvg/555464*
_output_shapes
:@2,
*batch_normalization_94/AssignMovingAvg/mulЛ
:batch_normalization_94/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_94_assignmovingavg_555464.batch_normalization_94/AssignMovingAvg/mul:z:06^batch_normalization_94/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_94/AssignMovingAvg/555464*
_output_shapes
 *
dtype02<
:batch_normalization_94/AssignMovingAvg/AssignSubVariableOpщ
.batch_normalization_94/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_94/AssignMovingAvg_1/555471*
_output_shapes
: *
dtype0*
valueB
 *  А?20
.batch_normalization_94/AssignMovingAvg_1/sub/xк
,batch_normalization_94/AssignMovingAvg_1/subSub7batch_normalization_94/AssignMovingAvg_1/sub/x:output:0'batch_normalization_94/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_94/AssignMovingAvg_1/555471*
_output_shapes
: 2.
,batch_normalization_94/AssignMovingAvg_1/sub▐
7batch_normalization_94/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_94_assignmovingavg_1_555471*
_output_shapes
:@*
dtype029
7batch_normalization_94/AssignMovingAvg_1/ReadVariableOp╦
.batch_normalization_94/AssignMovingAvg_1/sub_1Sub?batch_normalization_94/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_94/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_94/AssignMovingAvg_1/555471*
_output_shapes
:@20
.batch_normalization_94/AssignMovingAvg_1/sub_1▓
,batch_normalization_94/AssignMovingAvg_1/mulMul2batch_normalization_94/AssignMovingAvg_1/sub_1:z:00batch_normalization_94/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_94/AssignMovingAvg_1/555471*
_output_shapes
:@2.
,batch_normalization_94/AssignMovingAvg_1/mulЧ
<batch_normalization_94/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_94_assignmovingavg_1_5554710batch_normalization_94/AssignMovingAvg_1/mul:z:08^batch_normalization_94/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_94/AssignMovingAvg_1/555471*
_output_shapes
 *
dtype02>
<batch_normalization_94/AssignMovingAvg_1/AssignSubVariableOpЯ
leaky_re_lu_91/LeakyRelu	LeakyRelu+batch_normalization_94/FusedBatchNormV3:y:0*/
_output_shapes
:         @2
leaky_re_lu_91/LeakyReluw
dropout_36/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_36/dropout/rateК
dropout_36/dropout/ShapeShape&leaky_re_lu_91/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_36/dropout/ShapeУ
%dropout_36/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_36/dropout/random_uniform/minУ
%dropout_36/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2'
%dropout_36/dropout/random_uniform/max▌
/dropout_36/dropout/random_uniform/RandomUniformRandomUniform!dropout_36/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype021
/dropout_36/dropout/random_uniform/RandomUniform╓
%dropout_36/dropout/random_uniform/subSub.dropout_36/dropout/random_uniform/max:output:0.dropout_36/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_36/dropout/random_uniform/subЇ
%dropout_36/dropout/random_uniform/mulMul8dropout_36/dropout/random_uniform/RandomUniform:output:0)dropout_36/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         @2'
%dropout_36/dropout/random_uniform/mulт
!dropout_36/dropout/random_uniformAdd)dropout_36/dropout/random_uniform/mul:z:0.dropout_36/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         @2#
!dropout_36/dropout/random_uniformy
dropout_36/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout_36/dropout/sub/xЭ
dropout_36/dropout/subSub!dropout_36/dropout/sub/x:output:0 dropout_36/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_36/dropout/subБ
dropout_36/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout_36/dropout/truediv/xз
dropout_36/dropout/truedivRealDiv%dropout_36/dropout/truediv/x:output:0dropout_36/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_36/dropout/truediv╒
dropout_36/dropout/GreaterEqualGreaterEqual%dropout_36/dropout/random_uniform:z:0 dropout_36/dropout/rate:output:0*
T0*/
_output_shapes
:         @2!
dropout_36/dropout/GreaterEqual╣
dropout_36/dropout/mulMul&leaky_re_lu_91/LeakyRelu:activations:0dropout_36/dropout/truediv:z:0*
T0*/
_output_shapes
:         @2
dropout_36/dropout/mulи
dropout_36/dropout/CastCast#dropout_36/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout_36/dropout/Castо
dropout_36/dropout/mul_1Muldropout_36/dropout/mul:z:0dropout_36/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout_36/dropout/mul_1│
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_37/Conv2D/ReadVariableOp╫
conv2d_37/Conv2DConv2Ddropout_36/dropout/mul_1:z:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_37/Conv2Dк
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp░
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_37/BiasAddМ
#batch_normalization_95/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_95/LogicalAnd/xМ
#batch_normalization_95/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_95/LogicalAnd/y╚
!batch_normalization_95/LogicalAnd
LogicalAnd,batch_normalization_95/LogicalAnd/x:output:0,batch_normalization_95/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_95/LogicalAnd╣
%batch_normalization_95/ReadVariableOpReadVariableOp.batch_normalization_95_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_95/ReadVariableOp┐
'batch_normalization_95/ReadVariableOp_1ReadVariableOp0batch_normalization_95_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_95/ReadVariableOp_1
batch_normalization_95/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_95/ConstГ
batch_normalization_95/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_95/Const_1г
'batch_normalization_95/FusedBatchNormV3FusedBatchNormV3conv2d_37/BiasAdd:output:0-batch_normalization_95/ReadVariableOp:value:0/batch_normalization_95/ReadVariableOp_1:value:0%batch_normalization_95/Const:output:0'batch_normalization_95/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:2)
'batch_normalization_95/FusedBatchNormV3Е
batch_normalization_95/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2 
batch_normalization_95/Const_2у
,batch_normalization_95/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_95/AssignMovingAvg/555517*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,batch_normalization_95/AssignMovingAvg/sub/xв
*batch_normalization_95/AssignMovingAvg/subSub5batch_normalization_95/AssignMovingAvg/sub/x:output:0'batch_normalization_95/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_95/AssignMovingAvg/555517*
_output_shapes
: 2,
*batch_normalization_95/AssignMovingAvg/sub╪
5batch_normalization_95/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_95_assignmovingavg_555517*
_output_shapes
:@*
dtype027
5batch_normalization_95/AssignMovingAvg/ReadVariableOp┐
,batch_normalization_95/AssignMovingAvg/sub_1Sub=batch_normalization_95/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_95/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_95/AssignMovingAvg/555517*
_output_shapes
:@2.
,batch_normalization_95/AssignMovingAvg/sub_1и
*batch_normalization_95/AssignMovingAvg/mulMul0batch_normalization_95/AssignMovingAvg/sub_1:z:0.batch_normalization_95/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_95/AssignMovingAvg/555517*
_output_shapes
:@2,
*batch_normalization_95/AssignMovingAvg/mulЛ
:batch_normalization_95/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_95_assignmovingavg_555517.batch_normalization_95/AssignMovingAvg/mul:z:06^batch_normalization_95/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_95/AssignMovingAvg/555517*
_output_shapes
 *
dtype02<
:batch_normalization_95/AssignMovingAvg/AssignSubVariableOpщ
.batch_normalization_95/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_95/AssignMovingAvg_1/555524*
_output_shapes
: *
dtype0*
valueB
 *  А?20
.batch_normalization_95/AssignMovingAvg_1/sub/xк
,batch_normalization_95/AssignMovingAvg_1/subSub7batch_normalization_95/AssignMovingAvg_1/sub/x:output:0'batch_normalization_95/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_95/AssignMovingAvg_1/555524*
_output_shapes
: 2.
,batch_normalization_95/AssignMovingAvg_1/sub▐
7batch_normalization_95/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_95_assignmovingavg_1_555524*
_output_shapes
:@*
dtype029
7batch_normalization_95/AssignMovingAvg_1/ReadVariableOp╦
.batch_normalization_95/AssignMovingAvg_1/sub_1Sub?batch_normalization_95/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_95/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_95/AssignMovingAvg_1/555524*
_output_shapes
:@20
.batch_normalization_95/AssignMovingAvg_1/sub_1▓
,batch_normalization_95/AssignMovingAvg_1/mulMul2batch_normalization_95/AssignMovingAvg_1/sub_1:z:00batch_normalization_95/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_95/AssignMovingAvg_1/555524*
_output_shapes
:@2.
,batch_normalization_95/AssignMovingAvg_1/mulЧ
<batch_normalization_95/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_95_assignmovingavg_1_5555240batch_normalization_95/AssignMovingAvg_1/mul:z:08^batch_normalization_95/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_95/AssignMovingAvg_1/555524*
_output_shapes
 *
dtype02>
<batch_normalization_95/AssignMovingAvg_1/AssignSubVariableOpЯ
leaky_re_lu_92/LeakyRelu	LeakyRelu+batch_normalization_95/FusedBatchNormV3:y:0*/
_output_shapes
:         @2
leaky_re_lu_92/LeakyReluw
dropout_37/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_37/dropout/rateК
dropout_37/dropout/ShapeShape&leaky_re_lu_92/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_37/dropout/ShapeУ
%dropout_37/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_37/dropout/random_uniform/minУ
%dropout_37/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2'
%dropout_37/dropout/random_uniform/max▌
/dropout_37/dropout/random_uniform/RandomUniformRandomUniform!dropout_37/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype021
/dropout_37/dropout/random_uniform/RandomUniform╓
%dropout_37/dropout/random_uniform/subSub.dropout_37/dropout/random_uniform/max:output:0.dropout_37/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_37/dropout/random_uniform/subЇ
%dropout_37/dropout/random_uniform/mulMul8dropout_37/dropout/random_uniform/RandomUniform:output:0)dropout_37/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         @2'
%dropout_37/dropout/random_uniform/mulт
!dropout_37/dropout/random_uniformAdd)dropout_37/dropout/random_uniform/mul:z:0.dropout_37/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         @2#
!dropout_37/dropout/random_uniformy
dropout_37/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout_37/dropout/sub/xЭ
dropout_37/dropout/subSub!dropout_37/dropout/sub/x:output:0 dropout_37/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_37/dropout/subБ
dropout_37/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout_37/dropout/truediv/xз
dropout_37/dropout/truedivRealDiv%dropout_37/dropout/truediv/x:output:0dropout_37/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_37/dropout/truediv╒
dropout_37/dropout/GreaterEqualGreaterEqual%dropout_37/dropout/random_uniform:z:0 dropout_37/dropout/rate:output:0*
T0*/
_output_shapes
:         @2!
dropout_37/dropout/GreaterEqual╣
dropout_37/dropout/mulMul&leaky_re_lu_92/LeakyRelu:activations:0dropout_37/dropout/truediv:z:0*
T0*/
_output_shapes
:         @2
dropout_37/dropout/mulи
dropout_37/dropout/CastCast#dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout_37/dropout/Castо
dropout_37/dropout/mul_1Muldropout_37/dropout/mul:z:0dropout_37/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout_37/dropout/mul_1┤
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02!
conv2d_38/Conv2D/ReadVariableOp╪
conv2d_38/Conv2DConv2Ddropout_37/dropout/mul_1:z:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_38/Conv2Dл
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp▒
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_38/BiasAddМ
#batch_normalization_96/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_96/LogicalAnd/xМ
#batch_normalization_96/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_96/LogicalAnd/y╚
!batch_normalization_96/LogicalAnd
LogicalAnd,batch_normalization_96/LogicalAnd/x:output:0,batch_normalization_96/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_96/LogicalAnd║
%batch_normalization_96/ReadVariableOpReadVariableOp.batch_normalization_96_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%batch_normalization_96/ReadVariableOp└
'batch_normalization_96/ReadVariableOp_1ReadVariableOp0batch_normalization_96_readvariableop_1_resource*
_output_shapes	
:А*
dtype02)
'batch_normalization_96/ReadVariableOp_1
batch_normalization_96/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_96/ConstГ
batch_normalization_96/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_96/Const_1и
'batch_normalization_96/FusedBatchNormV3FusedBatchNormV3conv2d_38/BiasAdd:output:0-batch_normalization_96/ReadVariableOp:value:0/batch_normalization_96/ReadVariableOp_1:value:0%batch_normalization_96/Const:output:0'batch_normalization_96/Const_1:output:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:2)
'batch_normalization_96/FusedBatchNormV3Е
batch_normalization_96/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2 
batch_normalization_96/Const_2у
,batch_normalization_96/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_96/AssignMovingAvg/555570*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,batch_normalization_96/AssignMovingAvg/sub/xв
*batch_normalization_96/AssignMovingAvg/subSub5batch_normalization_96/AssignMovingAvg/sub/x:output:0'batch_normalization_96/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_96/AssignMovingAvg/555570*
_output_shapes
: 2,
*batch_normalization_96/AssignMovingAvg/sub┘
5batch_normalization_96/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_96_assignmovingavg_555570*
_output_shapes	
:А*
dtype027
5batch_normalization_96/AssignMovingAvg/ReadVariableOp└
,batch_normalization_96/AssignMovingAvg/sub_1Sub=batch_normalization_96/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_96/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_96/AssignMovingAvg/555570*
_output_shapes	
:А2.
,batch_normalization_96/AssignMovingAvg/sub_1й
*batch_normalization_96/AssignMovingAvg/mulMul0batch_normalization_96/AssignMovingAvg/sub_1:z:0.batch_normalization_96/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_96/AssignMovingAvg/555570*
_output_shapes	
:А2,
*batch_normalization_96/AssignMovingAvg/mulЛ
:batch_normalization_96/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_96_assignmovingavg_555570.batch_normalization_96/AssignMovingAvg/mul:z:06^batch_normalization_96/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_96/AssignMovingAvg/555570*
_output_shapes
 *
dtype02<
:batch_normalization_96/AssignMovingAvg/AssignSubVariableOpщ
.batch_normalization_96/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_96/AssignMovingAvg_1/555577*
_output_shapes
: *
dtype0*
valueB
 *  А?20
.batch_normalization_96/AssignMovingAvg_1/sub/xк
,batch_normalization_96/AssignMovingAvg_1/subSub7batch_normalization_96/AssignMovingAvg_1/sub/x:output:0'batch_normalization_96/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_96/AssignMovingAvg_1/555577*
_output_shapes
: 2.
,batch_normalization_96/AssignMovingAvg_1/sub▀
7batch_normalization_96/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_96_assignmovingavg_1_555577*
_output_shapes	
:А*
dtype029
7batch_normalization_96/AssignMovingAvg_1/ReadVariableOp╠
.batch_normalization_96/AssignMovingAvg_1/sub_1Sub?batch_normalization_96/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_96/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_96/AssignMovingAvg_1/555577*
_output_shapes	
:А20
.batch_normalization_96/AssignMovingAvg_1/sub_1│
,batch_normalization_96/AssignMovingAvg_1/mulMul2batch_normalization_96/AssignMovingAvg_1/sub_1:z:00batch_normalization_96/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_96/AssignMovingAvg_1/555577*
_output_shapes	
:А2.
,batch_normalization_96/AssignMovingAvg_1/mulЧ
<batch_normalization_96/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_96_assignmovingavg_1_5555770batch_normalization_96/AssignMovingAvg_1/mul:z:08^batch_normalization_96/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_96/AssignMovingAvg_1/555577*
_output_shapes
 *
dtype02>
<batch_normalization_96/AssignMovingAvg_1/AssignSubVariableOpа
leaky_re_lu_93/LeakyRelu	LeakyRelu+batch_normalization_96/FusedBatchNormV3:y:0*0
_output_shapes
:         А2
leaky_re_lu_93/LeakyReluw
dropout_38/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout_38/dropout/rateК
dropout_38/dropout/ShapeShape&leaky_re_lu_93/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_38/dropout/ShapeУ
%dropout_38/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_38/dropout/random_uniform/minУ
%dropout_38/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2'
%dropout_38/dropout/random_uniform/max▐
/dropout_38/dropout/random_uniform/RandomUniformRandomUniform!dropout_38/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype021
/dropout_38/dropout/random_uniform/RandomUniform╓
%dropout_38/dropout/random_uniform/subSub.dropout_38/dropout/random_uniform/max:output:0.dropout_38/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_38/dropout/random_uniform/subї
%dropout_38/dropout/random_uniform/mulMul8dropout_38/dropout/random_uniform/RandomUniform:output:0)dropout_38/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:         А2'
%dropout_38/dropout/random_uniform/mulу
!dropout_38/dropout/random_uniformAdd)dropout_38/dropout/random_uniform/mul:z:0.dropout_38/dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         А2#
!dropout_38/dropout/random_uniformy
dropout_38/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout_38/dropout/sub/xЭ
dropout_38/dropout/subSub!dropout_38/dropout/sub/x:output:0 dropout_38/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_38/dropout/subБ
dropout_38/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout_38/dropout/truediv/xз
dropout_38/dropout/truedivRealDiv%dropout_38/dropout/truediv/x:output:0dropout_38/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_38/dropout/truediv╓
dropout_38/dropout/GreaterEqualGreaterEqual%dropout_38/dropout/random_uniform:z:0 dropout_38/dropout/rate:output:0*
T0*0
_output_shapes
:         А2!
dropout_38/dropout/GreaterEqual║
dropout_38/dropout/mulMul&leaky_re_lu_93/LeakyRelu:activations:0dropout_38/dropout/truediv:z:0*
T0*0
_output_shapes
:         А2
dropout_38/dropout/mulй
dropout_38/dropout/CastCast#dropout_38/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout_38/dropout/Castп
dropout_38/dropout/mul_1Muldropout_38/dropout/mul:z:0dropout_38/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout_38/dropout/mul_1u
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  2
flatten_12/ConstЯ
flatten_12/ReshapeReshapedropout_38/dropout/mul_1:z:0flatten_12/Const:output:0*
T0*(
_output_shapes
:         А12
flatten_12/Reshapeй
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes
:	А1*
dtype02 
dense_43/MatMul/ReadVariableOpг
dense_43/MatMulMatMulflatten_12/Reshape:output:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_43/MatMulз
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_43/BiasAdd/ReadVariableOpе
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_43/BiasAddМ
#batch_normalization_97/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_97/LogicalAnd/xМ
#batch_normalization_97/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_97/LogicalAnd/y╚
!batch_normalization_97/LogicalAnd
LogicalAnd,batch_normalization_97/LogicalAnd/x:output:0,batch_normalization_97/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_97/LogicalAnd╕
5batch_normalization_97/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_97/moments/mean/reduction_indicesч
#batch_normalization_97/moments/meanMeandense_43/BiasAdd:output:0>batch_normalization_97/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_97/moments/mean┴
+batch_normalization_97/moments/StopGradientStopGradient,batch_normalization_97/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_97/moments/StopGradient№
0batch_normalization_97/moments/SquaredDifferenceSquaredDifferencedense_43/BiasAdd:output:04batch_normalization_97/moments/StopGradient:output:0*
T0*'
_output_shapes
:         22
0batch_normalization_97/moments/SquaredDifference└
9batch_normalization_97/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_97/moments/variance/reduction_indicesО
'batch_normalization_97/moments/varianceMean4batch_normalization_97/moments/SquaredDifference:z:0Bbatch_normalization_97/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_97/moments/variance┼
&batch_normalization_97/moments/SqueezeSqueeze,batch_normalization_97/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_97/moments/Squeeze═
(batch_normalization_97/moments/Squeeze_1Squeeze0batch_normalization_97/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_97/moments/Squeeze_1у
,batch_normalization_97/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_97/AssignMovingAvg/555620*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2.
,batch_normalization_97/AssignMovingAvg/decay╪
5batch_normalization_97/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_97_assignmovingavg_555620*
_output_shapes
:*
dtype027
5batch_normalization_97/AssignMovingAvg/ReadVariableOp╢
*batch_normalization_97/AssignMovingAvg/subSub=batch_normalization_97/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_97/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_97/AssignMovingAvg/555620*
_output_shapes
:2,
*batch_normalization_97/AssignMovingAvg/subн
*batch_normalization_97/AssignMovingAvg/mulMul.batch_normalization_97/AssignMovingAvg/sub:z:05batch_normalization_97/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_97/AssignMovingAvg/555620*
_output_shapes
:2,
*batch_normalization_97/AssignMovingAvg/mulЛ
:batch_normalization_97/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_97_assignmovingavg_555620.batch_normalization_97/AssignMovingAvg/mul:z:06^batch_normalization_97/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_97/AssignMovingAvg/555620*
_output_shapes
 *
dtype02<
:batch_normalization_97/AssignMovingAvg/AssignSubVariableOpщ
.batch_normalization_97/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_97/AssignMovingAvg_1/555626*
_output_shapes
: *
dtype0*
valueB
 *
╫#<20
.batch_normalization_97/AssignMovingAvg_1/decay▐
7batch_normalization_97/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_97_assignmovingavg_1_555626*
_output_shapes
:*
dtype029
7batch_normalization_97/AssignMovingAvg_1/ReadVariableOp└
,batch_normalization_97/AssignMovingAvg_1/subSub?batch_normalization_97/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_97/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_97/AssignMovingAvg_1/555626*
_output_shapes
:2.
,batch_normalization_97/AssignMovingAvg_1/sub╖
,batch_normalization_97/AssignMovingAvg_1/mulMul0batch_normalization_97/AssignMovingAvg_1/sub:z:07batch_normalization_97/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_97/AssignMovingAvg_1/555626*
_output_shapes
:2.
,batch_normalization_97/AssignMovingAvg_1/mulЧ
<batch_normalization_97/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_97_assignmovingavg_1_5556260batch_normalization_97/AssignMovingAvg_1/mul:z:08^batch_normalization_97/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_97/AssignMovingAvg_1/555626*
_output_shapes
 *
dtype02>
<batch_normalization_97/AssignMovingAvg_1/AssignSubVariableOp╚
*batch_normalization_97/Cast/ReadVariableOpReadVariableOp3batch_normalization_97_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_97/Cast/ReadVariableOp╬
,batch_normalization_97/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_97_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_97/Cast_1/ReadVariableOpХ
&batch_normalization_97/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2(
&batch_normalization_97/batchnorm/add/y▐
$batch_normalization_97/batchnorm/addAddV21batch_normalization_97/moments/Squeeze_1:output:0/batch_normalization_97/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_97/batchnorm/addи
&batch_normalization_97/batchnorm/RsqrtRsqrt(batch_normalization_97/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_97/batchnorm/Rsqrt┌
$batch_normalization_97/batchnorm/mulMul*batch_normalization_97/batchnorm/Rsqrt:y:04batch_normalization_97/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_97/batchnorm/mul╬
&batch_normalization_97/batchnorm/mul_1Muldense_43/BiasAdd:output:0(batch_normalization_97/batchnorm/mul:z:0*
T0*'
_output_shapes
:         2(
&batch_normalization_97/batchnorm/mul_1╫
&batch_normalization_97/batchnorm/mul_2Mul/batch_normalization_97/moments/Squeeze:output:0(batch_normalization_97/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_97/batchnorm/mul_2╪
$batch_normalization_97/batchnorm/subSub2batch_normalization_97/Cast/ReadVariableOp:value:0*batch_normalization_97/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_97/batchnorm/subс
&batch_normalization_97/batchnorm/add_1AddV2*batch_normalization_97/batchnorm/mul_1:z:0(batch_normalization_97/batchnorm/sub:z:0*
T0*'
_output_shapes
:         2(
&batch_normalization_97/batchnorm/add_1Ц
leaky_re_lu_94/LeakyRelu	LeakyRelu*batch_normalization_97/batchnorm/add_1:z:0*'
_output_shapes
:         2
leaky_re_lu_94/LeakyReluи
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_44/MatMul/ReadVariableOpо
dense_44/MatMulMatMul&leaky_re_lu_94/LeakyRelu:activations:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/MatMulз
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOpе
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_44/BiasAdd|
dense_44/SigmoidSigmoiddense_44/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_44/Sigmoid╟
IdentityIdentitydense_44/Sigmoid:y:0;^batch_normalization_94/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_94/AssignMovingAvg/ReadVariableOp=^batch_normalization_94/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_94/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_94/ReadVariableOp(^batch_normalization_94/ReadVariableOp_1;^batch_normalization_95/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_95/AssignMovingAvg/ReadVariableOp=^batch_normalization_95/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_95/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_95/ReadVariableOp(^batch_normalization_95/ReadVariableOp_1;^batch_normalization_96/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_96/AssignMovingAvg/ReadVariableOp=^batch_normalization_96/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_96/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_96/ReadVariableOp(^batch_normalization_96/ReadVariableOp_1;^batch_normalization_97/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_97/AssignMovingAvg/ReadVariableOp=^batch_normalization_97/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_97/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_97/Cast/ReadVariableOp-^batch_normalization_97/Cast_1/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::2x
:batch_normalization_94/AssignMovingAvg/AssignSubVariableOp:batch_normalization_94/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_94/AssignMovingAvg/ReadVariableOp5batch_normalization_94/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_94/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_94/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_94/AssignMovingAvg_1/ReadVariableOp7batch_normalization_94/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_94/ReadVariableOp%batch_normalization_94/ReadVariableOp2R
'batch_normalization_94/ReadVariableOp_1'batch_normalization_94/ReadVariableOp_12x
:batch_normalization_95/AssignMovingAvg/AssignSubVariableOp:batch_normalization_95/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_95/AssignMovingAvg/ReadVariableOp5batch_normalization_95/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_95/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_95/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_95/AssignMovingAvg_1/ReadVariableOp7batch_normalization_95/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_95/ReadVariableOp%batch_normalization_95/ReadVariableOp2R
'batch_normalization_95/ReadVariableOp_1'batch_normalization_95/ReadVariableOp_12x
:batch_normalization_96/AssignMovingAvg/AssignSubVariableOp:batch_normalization_96/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_96/AssignMovingAvg/ReadVariableOp5batch_normalization_96/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_96/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_96/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_96/AssignMovingAvg_1/ReadVariableOp7batch_normalization_96/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_96/ReadVariableOp%batch_normalization_96/ReadVariableOp2R
'batch_normalization_96/ReadVariableOp_1'batch_normalization_96/ReadVariableOp_12x
:batch_normalization_97/AssignMovingAvg/AssignSubVariableOp:batch_normalization_97/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_97/AssignMovingAvg/ReadVariableOp5batch_normalization_97/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_97/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_97/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_97/AssignMovingAvg_1/ReadVariableOp7batch_normalization_97/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_97/Cast/ReadVariableOp*batch_normalization_97/Cast/ReadVariableOp2\
,batch_normalization_97/Cast_1/ReadVariableOp,batch_normalization_97/Cast_1/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
В
ї
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_554399

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
ї
к
)__inference_dense_44_layer_call_fn_556339

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_44_layer_call_and_return_conditional_losses_5551792
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ъ
f
J__inference_leaky_re_lu_94_layer_call_and_return_conditional_losses_556316

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
г
А
7__inference_batch_normalization_97_layer_call_fn_556311

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_5548472
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
я
d
F__inference_dropout_36_layer_call_and_return_conditional_losses_554923

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+                           @2

IdentityГ

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+                           @:& "
 
_user_specified_nameinputs
┼
л
*__inference_conv2d_37_layer_call_fn_554426

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_37_layer_call_and_return_conditional_losses_5544182
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
т
K
/__inference_leaky_re_lu_94_layer_call_fn_556321

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_94_layer_call_and_return_conditional_losses_5551602
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
╩$
Э
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_554672

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_554657
assignmovingavg_1_554664
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
loc:@AssignMovingAvg/554657*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xп
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/554657*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_554657*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/554657*
_output_shapes	
:А2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/554657*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_554657AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/554657*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/554664*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╖
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/554664*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_554664*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/554664*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/554664*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_554664AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/554664*
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
Ш
e
F__inference_dropout_38_layer_call_and_return_conditional_losses_556182

inputs
identityИa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/random_uniform/max╧
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*B
_output_shapes0
.:,                           А*
dtype02&
$dropout/random_uniform/RandomUniformк
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub█
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*B
_output_shapes0
.:,                           А2
dropout/random_uniform/mul╔
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*B
_output_shapes0
.:,                           А2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv╝
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*B
_output_shapes0
.:,                           А2
dropout/GreaterEqualЛ
dropout/mulMulinputsdropout/truediv:z:0*
T0*B
_output_shapes0
.:,                           А2
dropout/mulЪ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*B
_output_shapes0
.:,                           А2
dropout/CastХ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*B
_output_shapes0
.:,                           А2
dropout/mul_1А
IdentityIdentitydropout/mul_1:z:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:& "
 
_user_specified_nameinputs
З
G
+__inference_flatten_12_layer_call_fn_556214

inputs
identity║
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:                  *-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_flatten_12_layer_call_and_return_conditional_losses_5551022
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:& "
 
_user_specified_nameinputs
├b
┼
I__inference_sequential_24_layer_call_and_return_conditional_losses_555291

inputs,
(conv2d_36_statefulpartitionedcall_args_1,
(conv2d_36_statefulpartitionedcall_args_29
5batch_normalization_94_statefulpartitionedcall_args_19
5batch_normalization_94_statefulpartitionedcall_args_29
5batch_normalization_94_statefulpartitionedcall_args_39
5batch_normalization_94_statefulpartitionedcall_args_4,
(conv2d_37_statefulpartitionedcall_args_1,
(conv2d_37_statefulpartitionedcall_args_29
5batch_normalization_95_statefulpartitionedcall_args_19
5batch_normalization_95_statefulpartitionedcall_args_29
5batch_normalization_95_statefulpartitionedcall_args_39
5batch_normalization_95_statefulpartitionedcall_args_4,
(conv2d_38_statefulpartitionedcall_args_1,
(conv2d_38_statefulpartitionedcall_args_29
5batch_normalization_96_statefulpartitionedcall_args_19
5batch_normalization_96_statefulpartitionedcall_args_29
5batch_normalization_96_statefulpartitionedcall_args_39
5batch_normalization_96_statefulpartitionedcall_args_4+
'dense_43_statefulpartitionedcall_args_1+
'dense_43_statefulpartitionedcall_args_29
5batch_normalization_97_statefulpartitionedcall_args_19
5batch_normalization_97_statefulpartitionedcall_args_29
5batch_normalization_97_statefulpartitionedcall_args_39
5batch_normalization_97_statefulpartitionedcall_args_4+
'dense_44_statefulpartitionedcall_args_1+
'dense_44_statefulpartitionedcall_args_2
identityИв.batch_normalization_94/StatefulPartitionedCallв.batch_normalization_95/StatefulPartitionedCallв.batch_normalization_96/StatefulPartitionedCallв.batch_normalization_97/StatefulPartitionedCallв!conv2d_36/StatefulPartitionedCallв!conv2d_37/StatefulPartitionedCallв!conv2d_38/StatefulPartitionedCallв dense_43/StatefulPartitionedCallв dense_44/StatefulPartitionedCallв"dropout_36/StatefulPartitionedCallв"dropout_37/StatefulPartitionedCallв"dropout_38/StatefulPartitionedCallє
 up_sampling2d_12/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_5542482"
 up_sampling2d_12/PartitionedCallя
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_12/PartitionedCall:output:0(conv2d_36_statefulpartitionedcall_args_1(conv2d_36_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_36_layer_call_and_return_conditional_losses_5542662#
!conv2d_36/StatefulPartitionedCallб
.batch_normalization_94/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:05batch_normalization_94_statefulpartitionedcall_args_15batch_normalization_94_statefulpartitionedcall_args_25batch_normalization_94_statefulpartitionedcall_args_35batch_normalization_94_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_55436820
.batch_normalization_94/StatefulPartitionedCallЮ
leaky_re_lu_91/PartitionedCallPartitionedCall7batch_normalization_94/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_91_layer_call_and_return_conditional_losses_5548902 
leaky_re_lu_91/PartitionedCallЪ
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_91/PartitionedCall:output:0*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_5549182$
"dropout_36/StatefulPartitionedCallё
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0(conv2d_37_statefulpartitionedcall_args_1(conv2d_37_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_37_layer_call_and_return_conditional_losses_5544182#
!conv2d_37/StatefulPartitionedCallб
.batch_normalization_95/StatefulPartitionedCallStatefulPartitionedCall*conv2d_37/StatefulPartitionedCall:output:05batch_normalization_95_statefulpartitionedcall_args_15batch_normalization_95_statefulpartitionedcall_args_25batch_normalization_95_statefulpartitionedcall_args_35batch_normalization_95_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_55452020
.batch_normalization_95/StatefulPartitionedCallЮ
leaky_re_lu_92/PartitionedCallPartitionedCall7batch_normalization_95/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_92_layer_call_and_return_conditional_losses_5549672 
leaky_re_lu_92/PartitionedCall┐
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_92/PartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_5549952$
"dropout_37/StatefulPartitionedCallЄ
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0(conv2d_38_statefulpartitionedcall_args_1(conv2d_38_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_38_layer_call_and_return_conditional_losses_5545702#
!conv2d_38/StatefulPartitionedCallв
.batch_normalization_96/StatefulPartitionedCallStatefulPartitionedCall*conv2d_38/StatefulPartitionedCall:output:05batch_normalization_96_statefulpartitionedcall_args_15batch_normalization_96_statefulpartitionedcall_args_25batch_normalization_96_statefulpartitionedcall_args_35batch_normalization_96_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_55467220
.batch_normalization_96/StatefulPartitionedCallЯ
leaky_re_lu_93/PartitionedCallPartitionedCall7batch_normalization_96/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_93_layer_call_and_return_conditional_losses_5550442 
leaky_re_lu_93/PartitionedCall└
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_93/PartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_5550722$
"dropout_38/StatefulPartitionedCallї
flatten_12/PartitionedCallPartitionedCall+dropout_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:                  *-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_flatten_12_layer_call_and_return_conditional_losses_5551022
flatten_12/PartitionedCall╩
 dense_43/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0'dense_43_statefulpartitionedcall_args_1'dense_43_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_5551202"
 dense_43/StatefulPartitionedCallЖ
.batch_normalization_97/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:05batch_normalization_97_statefulpartitionedcall_args_15batch_normalization_97_statefulpartitionedcall_args_25batch_normalization_97_statefulpartitionedcall_args_35batch_normalization_97_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_55481520
.batch_normalization_97/StatefulPartitionedCallД
leaky_re_lu_94/PartitionedCallPartitionedCall7batch_normalization_97/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_94_layer_call_and_return_conditional_losses_5551602 
leaky_re_lu_94/PartitionedCall╬
 dense_44/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_94/PartitionedCall:output:0'dense_44_statefulpartitionedcall_args_1'dense_44_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_44_layer_call_and_return_conditional_losses_5551792"
 dense_44/StatefulPartitionedCallт
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0/^batch_normalization_94/StatefulPartitionedCall/^batch_normalization_95/StatefulPartitionedCall/^batch_normalization_96/StatefulPartitionedCall/^batch_normalization_97/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::2`
.batch_normalization_94/StatefulPartitionedCall.batch_normalization_94/StatefulPartitionedCall2`
.batch_normalization_95/StatefulPartitionedCall.batch_normalization_95/StatefulPartitionedCall2`
.batch_normalization_96/StatefulPartitionedCall.batch_normalization_96/StatefulPartitionedCall2`
.batch_normalization_97/StatefulPartitionedCall.batch_normalization_97/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
п.
▒
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_554815

inputs
assignmovingavg_554790
assignmovingavg_1_554796 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвCast/ReadVariableOpвCast_1/ReadVariableOp^
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
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/554790*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_554790*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp├
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/554790*
_output_shapes
:2
AssignMovingAvg/sub║
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/554790*
_output_shapes
:2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_554790AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/554790*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/554796*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_554796*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp═
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/554796*
_output_shapes
:2
AssignMovingAvg_1/sub─
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/554796*
_output_shapes
:2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_554796AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/554796*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1з
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:& "
 
_user_specified_nameinputs
О
e
F__inference_dropout_36_layer_call_and_return_conditional_losses_555944

inputs
identityИa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/random_uniform/max╬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*A
_output_shapes/
-:+                           @*
dtype02&
$dropout/random_uniform/RandomUniformк
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub┌
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*A
_output_shapes/
-:+                           @2
dropout/random_uniform/mul╚
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv╗
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/GreaterEqualК
dropout/mulMulinputsdropout/truediv:z:0*
T0*A
_output_shapes/
-:+                           @2
dropout/mulЩ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                           @2
dropout/CastФ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                           @2
dropout/mul_1
IdentityIdentitydropout/mul_1:z:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:& "
 
_user_specified_nameinputs
О
e
F__inference_dropout_37_layer_call_and_return_conditional_losses_556063

inputs
identityИa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/random_uniform/max╬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*A
_output_shapes/
-:+                           @*
dtype02&
$dropout/random_uniform/RandomUniformк
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub┌
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*A
_output_shapes/
-:+                           @2
dropout/random_uniform/mul╚
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv╗
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/GreaterEqualК
dropout/mulMulinputsdropout/truediv:z:0*
T0*A
_output_shapes/
-:+                           @2
dropout/mulЩ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                           @2
dropout/CastФ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                           @2
dropout/mul_1
IdentityIdentitydropout/mul_1:z:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:& "
 
_user_specified_nameinputs
щ
f
J__inference_leaky_re_lu_91_layer_call_and_return_conditional_losses_554890

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
я
d
F__inference_dropout_36_layer_call_and_return_conditional_losses_555949

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+                           @2

IdentityГ

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+                           @:& "
 
_user_specified_nameinputs
Н
ї
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_556134

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
╨
M
1__inference_up_sampling2d_12_layer_call_fn_554254

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_5542482
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
р]
╫
I__inference_sequential_24_layer_call_and_return_conditional_losses_555240
input_1,
(conv2d_36_statefulpartitionedcall_args_1,
(conv2d_36_statefulpartitionedcall_args_29
5batch_normalization_94_statefulpartitionedcall_args_19
5batch_normalization_94_statefulpartitionedcall_args_29
5batch_normalization_94_statefulpartitionedcall_args_39
5batch_normalization_94_statefulpartitionedcall_args_4,
(conv2d_37_statefulpartitionedcall_args_1,
(conv2d_37_statefulpartitionedcall_args_29
5batch_normalization_95_statefulpartitionedcall_args_19
5batch_normalization_95_statefulpartitionedcall_args_29
5batch_normalization_95_statefulpartitionedcall_args_39
5batch_normalization_95_statefulpartitionedcall_args_4,
(conv2d_38_statefulpartitionedcall_args_1,
(conv2d_38_statefulpartitionedcall_args_29
5batch_normalization_96_statefulpartitionedcall_args_19
5batch_normalization_96_statefulpartitionedcall_args_29
5batch_normalization_96_statefulpartitionedcall_args_39
5batch_normalization_96_statefulpartitionedcall_args_4+
'dense_43_statefulpartitionedcall_args_1+
'dense_43_statefulpartitionedcall_args_29
5batch_normalization_97_statefulpartitionedcall_args_19
5batch_normalization_97_statefulpartitionedcall_args_29
5batch_normalization_97_statefulpartitionedcall_args_39
5batch_normalization_97_statefulpartitionedcall_args_4+
'dense_44_statefulpartitionedcall_args_1+
'dense_44_statefulpartitionedcall_args_2
identityИв.batch_normalization_94/StatefulPartitionedCallв.batch_normalization_95/StatefulPartitionedCallв.batch_normalization_96/StatefulPartitionedCallв.batch_normalization_97/StatefulPartitionedCallв!conv2d_36/StatefulPartitionedCallв!conv2d_37/StatefulPartitionedCallв!conv2d_38/StatefulPartitionedCallв dense_43/StatefulPartitionedCallв dense_44/StatefulPartitionedCallЇ
 up_sampling2d_12/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_5542482"
 up_sampling2d_12/PartitionedCallя
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_12/PartitionedCall:output:0(conv2d_36_statefulpartitionedcall_args_1(conv2d_36_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_36_layer_call_and_return_conditional_losses_5542662#
!conv2d_36/StatefulPartitionedCallб
.batch_normalization_94/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:05batch_normalization_94_statefulpartitionedcall_args_15batch_normalization_94_statefulpartitionedcall_args_25batch_normalization_94_statefulpartitionedcall_args_35batch_normalization_94_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_55439920
.batch_normalization_94/StatefulPartitionedCallЮ
leaky_re_lu_91/PartitionedCallPartitionedCall7batch_normalization_94/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_91_layer_call_and_return_conditional_losses_5548902 
leaky_re_lu_91/PartitionedCallВ
dropout_36/PartitionedCallPartitionedCall'leaky_re_lu_91/PartitionedCall:output:0*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_5549232
dropout_36/PartitionedCallщ
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0(conv2d_37_statefulpartitionedcall_args_1(conv2d_37_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_37_layer_call_and_return_conditional_losses_5544182#
!conv2d_37/StatefulPartitionedCallб
.batch_normalization_95/StatefulPartitionedCallStatefulPartitionedCall*conv2d_37/StatefulPartitionedCall:output:05batch_normalization_95_statefulpartitionedcall_args_15batch_normalization_95_statefulpartitionedcall_args_25batch_normalization_95_statefulpartitionedcall_args_35batch_normalization_95_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_55455120
.batch_normalization_95/StatefulPartitionedCallЮ
leaky_re_lu_92/PartitionedCallPartitionedCall7batch_normalization_95/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_92_layer_call_and_return_conditional_losses_5549672 
leaky_re_lu_92/PartitionedCallВ
dropout_37/PartitionedCallPartitionedCall'leaky_re_lu_92/PartitionedCall:output:0*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_5550002
dropout_37/PartitionedCallъ
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0(conv2d_38_statefulpartitionedcall_args_1(conv2d_38_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_38_layer_call_and_return_conditional_losses_5545702#
!conv2d_38/StatefulPartitionedCallв
.batch_normalization_96/StatefulPartitionedCallStatefulPartitionedCall*conv2d_38/StatefulPartitionedCall:output:05batch_normalization_96_statefulpartitionedcall_args_15batch_normalization_96_statefulpartitionedcall_args_25batch_normalization_96_statefulpartitionedcall_args_35batch_normalization_96_statefulpartitionedcall_args_4*
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
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_55470320
.batch_normalization_96/StatefulPartitionedCallЯ
leaky_re_lu_93/PartitionedCallPartitionedCall7batch_normalization_96/StatefulPartitionedCall:output:0*
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
J__inference_leaky_re_lu_93_layer_call_and_return_conditional_losses_5550442 
leaky_re_lu_93/PartitionedCallГ
dropout_38/PartitionedCallPartitionedCall'leaky_re_lu_93/PartitionedCall:output:0*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_5550772
dropout_38/PartitionedCallэ
flatten_12/PartitionedCallPartitionedCall#dropout_38/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:                  *-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_flatten_12_layer_call_and_return_conditional_losses_5551022
flatten_12/PartitionedCall╩
 dense_43/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0'dense_43_statefulpartitionedcall_args_1'dense_43_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_43_layer_call_and_return_conditional_losses_5551202"
 dense_43/StatefulPartitionedCallЖ
.batch_normalization_97/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:05batch_normalization_97_statefulpartitionedcall_args_15batch_normalization_97_statefulpartitionedcall_args_25batch_normalization_97_statefulpartitionedcall_args_35batch_normalization_97_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_55484720
.batch_normalization_97/StatefulPartitionedCallД
leaky_re_lu_94/PartitionedCallPartitionedCall7batch_normalization_97/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_leaky_re_lu_94_layer_call_and_return_conditional_losses_5551602 
leaky_re_lu_94/PartitionedCall╬
 dense_44/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_94/PartitionedCall:output:0'dense_44_statefulpartitionedcall_args_1'dense_44_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_44_layer_call_and_return_conditional_losses_5551792"
 dense_44/StatefulPartitionedCallє
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0/^batch_normalization_94/StatefulPartitionedCall/^batch_normalization_95/StatefulPartitionedCall/^batch_normalization_96/StatefulPartitionedCall/^batch_normalization_97/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::2`
.batch_normalization_94/StatefulPartitionedCall.batch_normalization_94/StatefulPartitionedCall2`
.batch_normalization_95/StatefulPartitionedCall.batch_normalization_95/StatefulPartitionedCall2`
.batch_normalization_96/StatefulPartitionedCall.batch_normalization_96/StatefulPartitionedCall2`
.batch_normalization_97/StatefulPartitionedCall.batch_normalization_97/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
▒
K
/__inference_leaky_re_lu_91_layer_call_fn_555924

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
J__inference_leaky_re_lu_91_layer_call_and_return_conditional_losses_5548902
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
╖
d
+__inference_dropout_38_layer_call_fn_556192

inputs
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_5550722
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
┤
K
/__inference_leaky_re_lu_93_layer_call_fn_556162

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
J__inference_leaky_re_lu_93_layer_call_and_return_conditional_losses_5550442
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
м
G
+__inference_dropout_38_layer_call_fn_556197

inputs
identity╠
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
CPU2*0J 8*O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_5550772
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
┤
d
+__inference_dropout_37_layer_call_fn_556073

inputs
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_5549952
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Є
d
F__inference_dropout_38_layer_call_and_return_conditional_losses_555077

inputs

identity_1u
IdentityIdentityinputs*
T0*B
_output_shapes0
.:,                           А2

IdentityД

Identity_1IdentityIdentity:output:0*
T0*B
_output_shapes0
.:,                           А2

Identity_1"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,                           А:& "
 
_user_specified_nameinputs
╟	
▌
D__inference_dense_44_layer_call_and_return_conditional_losses_556332

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ё
А
7__inference_batch_normalization_94_layer_call_fn_555914

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
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_5543992
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
ь
f
J__inference_leaky_re_lu_93_layer_call_and_return_conditional_losses_556157

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
С
▌
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_556293

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityИвCast/ReadVariableOpвCast_1/ReadVariableOpвCast_2/ReadVariableOpвCast_3/ReadVariableOp^
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

LogicalAndГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЕ
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         2
batchnorm/add_1┼
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:& "
 
_user_specified_nameinputs
╒
Ч
$__inference_signature_wrapper_555431
input_1"
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
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identityИвStatefulPartitionedCallР	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__wrapped_model_5542352
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
я
d
F__inference_dropout_37_layer_call_and_return_conditional_losses_556068

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+                           @2

IdentityГ

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+                           @:& "
 
_user_specified_nameinputs
Ї
А
7__inference_batch_normalization_96_layer_call_fn_556152

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
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_5547032
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
я
d
F__inference_dropout_37_layer_call_and_return_conditional_losses_555000

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+                           @2

IdentityГ

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+                           @:& "
 
_user_specified_nameinputs
╜
h
L__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_554248

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul╒
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborд
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
З
б
.__inference_sequential_24_layer_call_fn_555399
input_1"
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
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identityИвStatefulPartitionedCall╕	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_5553702
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
╟r
е
"__inference__traced_restore_556531
file_prefix3
/assignvariableop_sequential_24_conv2d_36_kernel3
/assignvariableop_1_sequential_24_conv2d_36_biasA
=assignvariableop_2_sequential_24_batch_normalization_94_gamma@
<assignvariableop_3_sequential_24_batch_normalization_94_betaG
Cassignvariableop_4_sequential_24_batch_normalization_94_moving_meanK
Gassignvariableop_5_sequential_24_batch_normalization_94_moving_variance5
1assignvariableop_6_sequential_24_conv2d_37_kernel3
/assignvariableop_7_sequential_24_conv2d_37_biasA
=assignvariableop_8_sequential_24_batch_normalization_95_gamma@
<assignvariableop_9_sequential_24_batch_normalization_95_betaH
Dassignvariableop_10_sequential_24_batch_normalization_95_moving_meanL
Hassignvariableop_11_sequential_24_batch_normalization_95_moving_variance6
2assignvariableop_12_sequential_24_conv2d_38_kernel4
0assignvariableop_13_sequential_24_conv2d_38_biasB
>assignvariableop_14_sequential_24_batch_normalization_96_gammaA
=assignvariableop_15_sequential_24_batch_normalization_96_betaH
Dassignvariableop_16_sequential_24_batch_normalization_96_moving_meanL
Hassignvariableop_17_sequential_24_batch_normalization_96_moving_variance5
1assignvariableop_18_sequential_24_dense_43_kernel3
/assignvariableop_19_sequential_24_dense_43_biasB
>assignvariableop_20_sequential_24_batch_normalization_97_gammaA
=assignvariableop_21_sequential_24_batch_normalization_97_betaH
Dassignvariableop_22_sequential_24_batch_normalization_97_moving_meanL
Hassignvariableop_23_sequential_24_batch_normalization_97_moving_variance5
1assignvariableop_24_sequential_24_dense_44_kernel3
/assignvariableop_25_sequential_24_dense_44_bias
identity_27ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1Х

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*б	
valueЧ	BФ	B)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB'layer-2/beta/.ATTRIBUTES/VARIABLE_VALUEB.layer-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2layer-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)layer-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-5/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB'layer-6/beta/.ATTRIBUTES/VARIABLE_VALUEB.layer-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2layer-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)layer-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB(layer-10/beta/.ATTRIBUTES/VARIABLE_VALUEB/layer-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3layer-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB*layer-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB(layer-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB(layer-15/beta/.ATTRIBUTES/VARIABLE_VALUEB/layer-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3layer-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB*layer-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB(layer-17/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names┬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesн
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp/assignvariableop_sequential_24_conv2d_36_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp/assignvariableop_1_sequential_24_conv2d_36_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2│
AssignVariableOp_2AssignVariableOp=assignvariableop_2_sequential_24_batch_normalization_94_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3▓
AssignVariableOp_3AssignVariableOp<assignvariableop_3_sequential_24_batch_normalization_94_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4╣
AssignVariableOp_4AssignVariableOpCassignvariableop_4_sequential_24_batch_normalization_94_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5╜
AssignVariableOp_5AssignVariableOpGassignvariableop_5_sequential_24_batch_normalization_94_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6з
AssignVariableOp_6AssignVariableOp1assignvariableop_6_sequential_24_conv2d_37_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7е
AssignVariableOp_7AssignVariableOp/assignvariableop_7_sequential_24_conv2d_37_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8│
AssignVariableOp_8AssignVariableOp=assignvariableop_8_sequential_24_batch_normalization_95_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9▓
AssignVariableOp_9AssignVariableOp<assignvariableop_9_sequential_24_batch_normalization_95_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10╜
AssignVariableOp_10AssignVariableOpDassignvariableop_10_sequential_24_batch_normalization_95_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11┴
AssignVariableOp_11AssignVariableOpHassignvariableop_11_sequential_24_batch_normalization_95_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12л
AssignVariableOp_12AssignVariableOp2assignvariableop_12_sequential_24_conv2d_38_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13й
AssignVariableOp_13AssignVariableOp0assignvariableop_13_sequential_24_conv2d_38_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14╖
AssignVariableOp_14AssignVariableOp>assignvariableop_14_sequential_24_batch_normalization_96_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15╢
AssignVariableOp_15AssignVariableOp=assignvariableop_15_sequential_24_batch_normalization_96_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16╜
AssignVariableOp_16AssignVariableOpDassignvariableop_16_sequential_24_batch_normalization_96_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17┴
AssignVariableOp_17AssignVariableOpHassignvariableop_17_sequential_24_batch_normalization_96_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18к
AssignVariableOp_18AssignVariableOp1assignvariableop_18_sequential_24_dense_43_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19и
AssignVariableOp_19AssignVariableOp/assignvariableop_19_sequential_24_dense_43_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20╖
AssignVariableOp_20AssignVariableOp>assignvariableop_20_sequential_24_batch_normalization_97_gammaIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21╢
AssignVariableOp_21AssignVariableOp=assignvariableop_21_sequential_24_batch_normalization_97_betaIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22╜
AssignVariableOp_22AssignVariableOpDassignvariableop_22_sequential_24_batch_normalization_97_moving_meanIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23┴
AssignVariableOp_23AssignVariableOpHassignvariableop_23_sequential_24_batch_normalization_97_moving_varianceIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24к
AssignVariableOp_24AssignVariableOp1assignvariableop_24_sequential_24_dense_44_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25и
AssignVariableOp_25AssignVariableOp/assignvariableop_25_sequential_24_dense_44_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25и
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
NoOpЪ
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26з
Identity_27IdentityIdentity_26:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_27"#
identity_27Identity_27:output:0*}
_input_shapesl
j: ::::::::::::::::::::::::::2$
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
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
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
╗$
Э
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_555993

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_555978
assignmovingavg_1_555985
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
loc:@AssignMovingAvg/555978*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xп
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/555978*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_555978*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp╠
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/555978*
_output_shapes
:@2
AssignMovingAvg/sub_1╡
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/555978*
_output_shapes
:@2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_555978AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/555978*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/555985*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╖
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/555985*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_555985*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╪
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/555985*
_output_shapes
:@2
AssignMovingAvg_1/sub_1┐
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/555985*
_output_shapes
:@2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_555985AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/555985*
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
г
А
7__inference_batch_normalization_97_layer_call_fn_556302

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*[
fVRT
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_5548152
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ы╒
У
!__inference__wrapped_model_554235
input_1:
6sequential_24_conv2d_36_conv2d_readvariableop_resource;
7sequential_24_conv2d_36_biasadd_readvariableop_resource@
<sequential_24_batch_normalization_94_readvariableop_resourceB
>sequential_24_batch_normalization_94_readvariableop_1_resourceQ
Msequential_24_batch_normalization_94_fusedbatchnormv3_readvariableop_resourceS
Osequential_24_batch_normalization_94_fusedbatchnormv3_readvariableop_1_resource:
6sequential_24_conv2d_37_conv2d_readvariableop_resource;
7sequential_24_conv2d_37_biasadd_readvariableop_resource@
<sequential_24_batch_normalization_95_readvariableop_resourceB
>sequential_24_batch_normalization_95_readvariableop_1_resourceQ
Msequential_24_batch_normalization_95_fusedbatchnormv3_readvariableop_resourceS
Osequential_24_batch_normalization_95_fusedbatchnormv3_readvariableop_1_resource:
6sequential_24_conv2d_38_conv2d_readvariableop_resource;
7sequential_24_conv2d_38_biasadd_readvariableop_resource@
<sequential_24_batch_normalization_96_readvariableop_resourceB
>sequential_24_batch_normalization_96_readvariableop_1_resourceQ
Msequential_24_batch_normalization_96_fusedbatchnormv3_readvariableop_resourceS
Osequential_24_batch_normalization_96_fusedbatchnormv3_readvariableop_1_resource9
5sequential_24_dense_43_matmul_readvariableop_resource:
6sequential_24_dense_43_biasadd_readvariableop_resourceE
Asequential_24_batch_normalization_97_cast_readvariableop_resourceG
Csequential_24_batch_normalization_97_cast_1_readvariableop_resourceG
Csequential_24_batch_normalization_97_cast_2_readvariableop_resourceG
Csequential_24_batch_normalization_97_cast_3_readvariableop_resource9
5sequential_24_dense_44_matmul_readvariableop_resource:
6sequential_24_dense_44_biasadd_readvariableop_resource
identityИвDsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOpвFsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOp_1в3sequential_24/batch_normalization_94/ReadVariableOpв5sequential_24/batch_normalization_94/ReadVariableOp_1вDsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOpвFsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOp_1в3sequential_24/batch_normalization_95/ReadVariableOpв5sequential_24/batch_normalization_95/ReadVariableOp_1вDsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOpвFsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOp_1в3sequential_24/batch_normalization_96/ReadVariableOpв5sequential_24/batch_normalization_96/ReadVariableOp_1в8sequential_24/batch_normalization_97/Cast/ReadVariableOpв:sequential_24/batch_normalization_97/Cast_1/ReadVariableOpв:sequential_24/batch_normalization_97/Cast_2/ReadVariableOpв:sequential_24/batch_normalization_97/Cast_3/ReadVariableOpв.sequential_24/conv2d_36/BiasAdd/ReadVariableOpв-sequential_24/conv2d_36/Conv2D/ReadVariableOpв.sequential_24/conv2d_37/BiasAdd/ReadVariableOpв-sequential_24/conv2d_37/Conv2D/ReadVariableOpв.sequential_24/conv2d_38/BiasAdd/ReadVariableOpв-sequential_24/conv2d_38/Conv2D/ReadVariableOpв-sequential_24/dense_43/BiasAdd/ReadVariableOpв,sequential_24/dense_43/MatMul/ReadVariableOpв-sequential_24/dense_44/BiasAdd/ReadVariableOpв,sequential_24/dense_44/MatMul/ReadVariableOpГ
$sequential_24/up_sampling2d_12/ShapeShapeinput_1*
T0*
_output_shapes
:2&
$sequential_24/up_sampling2d_12/Shape▓
2sequential_24/up_sampling2d_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2sequential_24/up_sampling2d_12/strided_slice/stack╢
4sequential_24/up_sampling2d_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_24/up_sampling2d_12/strided_slice/stack_1╢
4sequential_24/up_sampling2d_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_24/up_sampling2d_12/strided_slice/stack_2И
,sequential_24/up_sampling2d_12/strided_sliceStridedSlice-sequential_24/up_sampling2d_12/Shape:output:0;sequential_24/up_sampling2d_12/strided_slice/stack:output:0=sequential_24/up_sampling2d_12/strided_slice/stack_1:output:0=sequential_24/up_sampling2d_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2.
,sequential_24/up_sampling2d_12/strided_sliceЭ
$sequential_24/up_sampling2d_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2&
$sequential_24/up_sampling2d_12/Const┌
"sequential_24/up_sampling2d_12/mulMul5sequential_24/up_sampling2d_12/strided_slice:output:0-sequential_24/up_sampling2d_12/Const:output:0*
T0*
_output_shapes
:2$
"sequential_24/up_sampling2d_12/mulШ
;sequential_24/up_sampling2d_12/resize/ResizeNearestNeighborResizeNearestNeighborinput_1&sequential_24/up_sampling2d_12/mul:z:0*
T0*/
_output_shapes
:         88*
half_pixel_centers(2=
;sequential_24/up_sampling2d_12/resize/ResizeNearestNeighbor▌
-sequential_24/conv2d_36/Conv2D/ReadVariableOpReadVariableOp6sequential_24_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02/
-sequential_24/conv2d_36/Conv2D/ReadVariableOp▒
sequential_24/conv2d_36/Conv2DConv2DLsequential_24/up_sampling2d_12/resize/ResizeNearestNeighbor:resized_images:05sequential_24/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2 
sequential_24/conv2d_36/Conv2D╘
.sequential_24/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp7sequential_24_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_24/conv2d_36/BiasAdd/ReadVariableOpш
sequential_24/conv2d_36/BiasAddBiasAdd'sequential_24/conv2d_36/Conv2D:output:06sequential_24/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2!
sequential_24/conv2d_36/BiasAddи
1sequential_24/batch_normalization_94/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1sequential_24/batch_normalization_94/LogicalAnd/xи
1sequential_24/batch_normalization_94/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1sequential_24/batch_normalization_94/LogicalAnd/yА
/sequential_24/batch_normalization_94/LogicalAnd
LogicalAnd:sequential_24/batch_normalization_94/LogicalAnd/x:output:0:sequential_24/batch_normalization_94/LogicalAnd/y:output:0*
_output_shapes
: 21
/sequential_24/batch_normalization_94/LogicalAndу
3sequential_24/batch_normalization_94/ReadVariableOpReadVariableOp<sequential_24_batch_normalization_94_readvariableop_resource*
_output_shapes
:@*
dtype025
3sequential_24/batch_normalization_94/ReadVariableOpщ
5sequential_24/batch_normalization_94/ReadVariableOp_1ReadVariableOp>sequential_24_batch_normalization_94_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5sequential_24/batch_normalization_94/ReadVariableOp_1Ц
Dsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_24_batch_normalization_94_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_24_batch_normalization_94_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Fsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOp_1╩
5sequential_24/batch_normalization_94/FusedBatchNormV3FusedBatchNormV3(sequential_24/conv2d_36/BiasAdd:output:0;sequential_24/batch_normalization_94/ReadVariableOp:value:0=sequential_24/batch_normalization_94/ReadVariableOp_1:value:0Lsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 27
5sequential_24/batch_normalization_94/FusedBatchNormV3Э
*sequential_24/batch_normalization_94/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2,
*sequential_24/batch_normalization_94/Const╔
&sequential_24/leaky_re_lu_91/LeakyRelu	LeakyRelu9sequential_24/batch_normalization_94/FusedBatchNormV3:y:0*/
_output_shapes
:         @2(
&sequential_24/leaky_re_lu_91/LeakyRelu┬
!sequential_24/dropout_36/IdentityIdentity4sequential_24/leaky_re_lu_91/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @2#
!sequential_24/dropout_36/Identity▌
-sequential_24/conv2d_37/Conv2D/ReadVariableOpReadVariableOp6sequential_24_conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_24/conv2d_37/Conv2D/ReadVariableOpП
sequential_24/conv2d_37/Conv2DConv2D*sequential_24/dropout_36/Identity:output:05sequential_24/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2 
sequential_24/conv2d_37/Conv2D╘
.sequential_24/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp7sequential_24_conv2d_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_24/conv2d_37/BiasAdd/ReadVariableOpш
sequential_24/conv2d_37/BiasAddBiasAdd'sequential_24/conv2d_37/Conv2D:output:06sequential_24/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2!
sequential_24/conv2d_37/BiasAddи
1sequential_24/batch_normalization_95/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1sequential_24/batch_normalization_95/LogicalAnd/xи
1sequential_24/batch_normalization_95/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1sequential_24/batch_normalization_95/LogicalAnd/yА
/sequential_24/batch_normalization_95/LogicalAnd
LogicalAnd:sequential_24/batch_normalization_95/LogicalAnd/x:output:0:sequential_24/batch_normalization_95/LogicalAnd/y:output:0*
_output_shapes
: 21
/sequential_24/batch_normalization_95/LogicalAndу
3sequential_24/batch_normalization_95/ReadVariableOpReadVariableOp<sequential_24_batch_normalization_95_readvariableop_resource*
_output_shapes
:@*
dtype025
3sequential_24/batch_normalization_95/ReadVariableOpщ
5sequential_24/batch_normalization_95/ReadVariableOp_1ReadVariableOp>sequential_24_batch_normalization_95_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5sequential_24/batch_normalization_95/ReadVariableOp_1Ц
Dsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_24_batch_normalization_95_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOpЬ
Fsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_24_batch_normalization_95_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Fsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOp_1╩
5sequential_24/batch_normalization_95/FusedBatchNormV3FusedBatchNormV3(sequential_24/conv2d_37/BiasAdd:output:0;sequential_24/batch_normalization_95/ReadVariableOp:value:0=sequential_24/batch_normalization_95/ReadVariableOp_1:value:0Lsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 27
5sequential_24/batch_normalization_95/FusedBatchNormV3Э
*sequential_24/batch_normalization_95/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2,
*sequential_24/batch_normalization_95/Const╔
&sequential_24/leaky_re_lu_92/LeakyRelu	LeakyRelu9sequential_24/batch_normalization_95/FusedBatchNormV3:y:0*/
_output_shapes
:         @2(
&sequential_24/leaky_re_lu_92/LeakyRelu┬
!sequential_24/dropout_37/IdentityIdentity4sequential_24/leaky_re_lu_92/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @2#
!sequential_24/dropout_37/Identity▐
-sequential_24/conv2d_38/Conv2D/ReadVariableOpReadVariableOp6sequential_24_conv2d_38_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02/
-sequential_24/conv2d_38/Conv2D/ReadVariableOpР
sequential_24/conv2d_38/Conv2DConv2D*sequential_24/dropout_37/Identity:output:05sequential_24/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2 
sequential_24/conv2d_38/Conv2D╒
.sequential_24/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp7sequential_24_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.sequential_24/conv2d_38/BiasAdd/ReadVariableOpщ
sequential_24/conv2d_38/BiasAddBiasAdd'sequential_24/conv2d_38/Conv2D:output:06sequential_24/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2!
sequential_24/conv2d_38/BiasAddи
1sequential_24/batch_normalization_96/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1sequential_24/batch_normalization_96/LogicalAnd/xи
1sequential_24/batch_normalization_96/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1sequential_24/batch_normalization_96/LogicalAnd/yА
/sequential_24/batch_normalization_96/LogicalAnd
LogicalAnd:sequential_24/batch_normalization_96/LogicalAnd/x:output:0:sequential_24/batch_normalization_96/LogicalAnd/y:output:0*
_output_shapes
: 21
/sequential_24/batch_normalization_96/LogicalAndф
3sequential_24/batch_normalization_96/ReadVariableOpReadVariableOp<sequential_24_batch_normalization_96_readvariableop_resource*
_output_shapes	
:А*
dtype025
3sequential_24/batch_normalization_96/ReadVariableOpъ
5sequential_24/batch_normalization_96/ReadVariableOp_1ReadVariableOp>sequential_24_batch_normalization_96_readvariableop_1_resource*
_output_shapes	
:А*
dtype027
5sequential_24/batch_normalization_96/ReadVariableOp_1Ч
Dsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_24_batch_normalization_96_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02F
Dsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOpЭ
Fsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_24_batch_normalization_96_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02H
Fsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOp_1╧
5sequential_24/batch_normalization_96/FusedBatchNormV3FusedBatchNormV3(sequential_24/conv2d_38/BiasAdd:output:0;sequential_24/batch_normalization_96/ReadVariableOp:value:0=sequential_24/batch_normalization_96/ReadVariableOp_1:value:0Lsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 27
5sequential_24/batch_normalization_96/FusedBatchNormV3Э
*sequential_24/batch_normalization_96/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2,
*sequential_24/batch_normalization_96/Const╩
&sequential_24/leaky_re_lu_93/LeakyRelu	LeakyRelu9sequential_24/batch_normalization_96/FusedBatchNormV3:y:0*0
_output_shapes
:         А2(
&sequential_24/leaky_re_lu_93/LeakyRelu├
!sequential_24/dropout_38/IdentityIdentity4sequential_24/leaky_re_lu_93/LeakyRelu:activations:0*
T0*0
_output_shapes
:         А2#
!sequential_24/dropout_38/IdentityС
sequential_24/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  2 
sequential_24/flatten_12/Const╫
 sequential_24/flatten_12/ReshapeReshape*sequential_24/dropout_38/Identity:output:0'sequential_24/flatten_12/Const:output:0*
T0*(
_output_shapes
:         А12"
 sequential_24/flatten_12/Reshape╙
,sequential_24/dense_43/MatMul/ReadVariableOpReadVariableOp5sequential_24_dense_43_matmul_readvariableop_resource*
_output_shapes
:	А1*
dtype02.
,sequential_24/dense_43/MatMul/ReadVariableOp█
sequential_24/dense_43/MatMulMatMul)sequential_24/flatten_12/Reshape:output:04sequential_24/dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_24/dense_43/MatMul╤
-sequential_24/dense_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_24/dense_43/BiasAdd/ReadVariableOp▌
sequential_24/dense_43/BiasAddBiasAdd'sequential_24/dense_43/MatMul:product:05sequential_24/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
sequential_24/dense_43/BiasAddи
1sequential_24/batch_normalization_97/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1sequential_24/batch_normalization_97/LogicalAnd/xи
1sequential_24/batch_normalization_97/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1sequential_24/batch_normalization_97/LogicalAnd/yА
/sequential_24/batch_normalization_97/LogicalAnd
LogicalAnd:sequential_24/batch_normalization_97/LogicalAnd/x:output:0:sequential_24/batch_normalization_97/LogicalAnd/y:output:0*
_output_shapes
: 21
/sequential_24/batch_normalization_97/LogicalAndЄ
8sequential_24/batch_normalization_97/Cast/ReadVariableOpReadVariableOpAsequential_24_batch_normalization_97_cast_readvariableop_resource*
_output_shapes
:*
dtype02:
8sequential_24/batch_normalization_97/Cast/ReadVariableOp°
:sequential_24/batch_normalization_97/Cast_1/ReadVariableOpReadVariableOpCsequential_24_batch_normalization_97_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02<
:sequential_24/batch_normalization_97/Cast_1/ReadVariableOp°
:sequential_24/batch_normalization_97/Cast_2/ReadVariableOpReadVariableOpCsequential_24_batch_normalization_97_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02<
:sequential_24/batch_normalization_97/Cast_2/ReadVariableOp°
:sequential_24/batch_normalization_97/Cast_3/ReadVariableOpReadVariableOpCsequential_24_batch_normalization_97_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02<
:sequential_24/batch_normalization_97/Cast_3/ReadVariableOp▒
4sequential_24/batch_normalization_97/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:26
4sequential_24/batch_normalization_97/batchnorm/add/yЩ
2sequential_24/batch_normalization_97/batchnorm/addAddV2Bsequential_24/batch_normalization_97/Cast_1/ReadVariableOp:value:0=sequential_24/batch_normalization_97/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_24/batch_normalization_97/batchnorm/add╥
4sequential_24/batch_normalization_97/batchnorm/RsqrtRsqrt6sequential_24/batch_normalization_97/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_24/batch_normalization_97/batchnorm/RsqrtТ
2sequential_24/batch_normalization_97/batchnorm/mulMul8sequential_24/batch_normalization_97/batchnorm/Rsqrt:y:0Bsequential_24/batch_normalization_97/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_24/batch_normalization_97/batchnorm/mulЖ
4sequential_24/batch_normalization_97/batchnorm/mul_1Mul'sequential_24/dense_43/BiasAdd:output:06sequential_24/batch_normalization_97/batchnorm/mul:z:0*
T0*'
_output_shapes
:         26
4sequential_24/batch_normalization_97/batchnorm/mul_1Т
4sequential_24/batch_normalization_97/batchnorm/mul_2Mul@sequential_24/batch_normalization_97/Cast/ReadVariableOp:value:06sequential_24/batch_normalization_97/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_24/batch_normalization_97/batchnorm/mul_2Т
2sequential_24/batch_normalization_97/batchnorm/subSubBsequential_24/batch_normalization_97/Cast_2/ReadVariableOp:value:08sequential_24/batch_normalization_97/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_24/batch_normalization_97/batchnorm/subЩ
4sequential_24/batch_normalization_97/batchnorm/add_1AddV28sequential_24/batch_normalization_97/batchnorm/mul_1:z:06sequential_24/batch_normalization_97/batchnorm/sub:z:0*
T0*'
_output_shapes
:         26
4sequential_24/batch_normalization_97/batchnorm/add_1└
&sequential_24/leaky_re_lu_94/LeakyRelu	LeakyRelu8sequential_24/batch_normalization_97/batchnorm/add_1:z:0*'
_output_shapes
:         2(
&sequential_24/leaky_re_lu_94/LeakyRelu╥
,sequential_24/dense_44/MatMul/ReadVariableOpReadVariableOp5sequential_24_dense_44_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_24/dense_44/MatMul/ReadVariableOpц
sequential_24/dense_44/MatMulMatMul4sequential_24/leaky_re_lu_94/LeakyRelu:activations:04sequential_24/dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_24/dense_44/MatMul╤
-sequential_24/dense_44/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_24/dense_44/BiasAdd/ReadVariableOp▌
sequential_24/dense_44/BiasAddBiasAdd'sequential_24/dense_44/MatMul:product:05sequential_24/dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
sequential_24/dense_44/BiasAddж
sequential_24/dense_44/SigmoidSigmoid'sequential_24/dense_44/BiasAdd:output:0*
T0*'
_output_shapes
:         2 
sequential_24/dense_44/Sigmoid├
IdentityIdentity"sequential_24/dense_44/Sigmoid:y:0E^sequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOpG^sequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOp_14^sequential_24/batch_normalization_94/ReadVariableOp6^sequential_24/batch_normalization_94/ReadVariableOp_1E^sequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOpG^sequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOp_14^sequential_24/batch_normalization_95/ReadVariableOp6^sequential_24/batch_normalization_95/ReadVariableOp_1E^sequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOpG^sequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOp_14^sequential_24/batch_normalization_96/ReadVariableOp6^sequential_24/batch_normalization_96/ReadVariableOp_19^sequential_24/batch_normalization_97/Cast/ReadVariableOp;^sequential_24/batch_normalization_97/Cast_1/ReadVariableOp;^sequential_24/batch_normalization_97/Cast_2/ReadVariableOp;^sequential_24/batch_normalization_97/Cast_3/ReadVariableOp/^sequential_24/conv2d_36/BiasAdd/ReadVariableOp.^sequential_24/conv2d_36/Conv2D/ReadVariableOp/^sequential_24/conv2d_37/BiasAdd/ReadVariableOp.^sequential_24/conv2d_37/Conv2D/ReadVariableOp/^sequential_24/conv2d_38/BiasAdd/ReadVariableOp.^sequential_24/conv2d_38/Conv2D/ReadVariableOp.^sequential_24/dense_43/BiasAdd/ReadVariableOp-^sequential_24/dense_43/MatMul/ReadVariableOp.^sequential_24/dense_44/BiasAdd/ReadVariableOp-^sequential_24/dense_44/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::2М
Dsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOpDsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOp_1Fsequential_24/batch_normalization_94/FusedBatchNormV3/ReadVariableOp_12j
3sequential_24/batch_normalization_94/ReadVariableOp3sequential_24/batch_normalization_94/ReadVariableOp2n
5sequential_24/batch_normalization_94/ReadVariableOp_15sequential_24/batch_normalization_94/ReadVariableOp_12М
Dsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOpDsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOp_1Fsequential_24/batch_normalization_95/FusedBatchNormV3/ReadVariableOp_12j
3sequential_24/batch_normalization_95/ReadVariableOp3sequential_24/batch_normalization_95/ReadVariableOp2n
5sequential_24/batch_normalization_95/ReadVariableOp_15sequential_24/batch_normalization_95/ReadVariableOp_12М
Dsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOpDsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOp_1Fsequential_24/batch_normalization_96/FusedBatchNormV3/ReadVariableOp_12j
3sequential_24/batch_normalization_96/ReadVariableOp3sequential_24/batch_normalization_96/ReadVariableOp2n
5sequential_24/batch_normalization_96/ReadVariableOp_15sequential_24/batch_normalization_96/ReadVariableOp_12t
8sequential_24/batch_normalization_97/Cast/ReadVariableOp8sequential_24/batch_normalization_97/Cast/ReadVariableOp2x
:sequential_24/batch_normalization_97/Cast_1/ReadVariableOp:sequential_24/batch_normalization_97/Cast_1/ReadVariableOp2x
:sequential_24/batch_normalization_97/Cast_2/ReadVariableOp:sequential_24/batch_normalization_97/Cast_2/ReadVariableOp2x
:sequential_24/batch_normalization_97/Cast_3/ReadVariableOp:sequential_24/batch_normalization_97/Cast_3/ReadVariableOp2`
.sequential_24/conv2d_36/BiasAdd/ReadVariableOp.sequential_24/conv2d_36/BiasAdd/ReadVariableOp2^
-sequential_24/conv2d_36/Conv2D/ReadVariableOp-sequential_24/conv2d_36/Conv2D/ReadVariableOp2`
.sequential_24/conv2d_37/BiasAdd/ReadVariableOp.sequential_24/conv2d_37/BiasAdd/ReadVariableOp2^
-sequential_24/conv2d_37/Conv2D/ReadVariableOp-sequential_24/conv2d_37/Conv2D/ReadVariableOp2`
.sequential_24/conv2d_38/BiasAdd/ReadVariableOp.sequential_24/conv2d_38/BiasAdd/ReadVariableOp2^
-sequential_24/conv2d_38/Conv2D/ReadVariableOp-sequential_24/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_24/dense_43/BiasAdd/ReadVariableOp-sequential_24/dense_43/BiasAdd/ReadVariableOp2\
,sequential_24/dense_43/MatMul/ReadVariableOp,sequential_24/dense_43/MatMul/ReadVariableOp2^
-sequential_24/dense_44/BiasAdd/ReadVariableOp-sequential_24/dense_44/BiasAdd/ReadVariableOp2\
,sequential_24/dense_44/MatMul/ReadVariableOp,sequential_24/dense_44/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1
╟	
▌
D__inference_dense_44_layer_call_and_return_conditional_losses_555179

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
й
G
+__inference_dropout_36_layer_call_fn_555959

inputs
identity╦
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
CPU2*0J 8*O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_5549232
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
О
e
F__inference_dropout_36_layer_call_and_return_conditional_losses_554918

inputs
identityИa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/random_uniform/max╬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*A
_output_shapes/
-:+                           @*
dtype02&
$dropout/random_uniform/RandomUniformк
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub┌
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*A
_output_shapes/
-:+                           @2
dropout/random_uniform/mul╚
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv╗
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/GreaterEqualК
dropout/mulMulinputsdropout/truediv:z:0*
T0*A
_output_shapes/
-:+                           @2
dropout/mulЩ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                           @2
dropout/CastФ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                           @2
dropout/mul_1
IdentityIdentitydropout/mul_1:z:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:& "
 
_user_specified_nameinputs
є

▐
E__inference_conv2d_37_layer_call_and_return_conditional_losses_554418

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddп
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Ш
e
F__inference_dropout_38_layer_call_and_return_conditional_losses_555072

inputs
identityИa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/random_uniform/max╧
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*B
_output_shapes0
.:,                           А*
dtype02&
$dropout/random_uniform/RandomUniformк
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub█
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*B
_output_shapes0
.:,                           А2
dropout/random_uniform/mul╔
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*B
_output_shapes0
.:,                           А2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv╝
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*B
_output_shapes0
.:,                           А2
dropout/GreaterEqualЛ
dropout/mulMulinputsdropout/truediv:z:0*
T0*B
_output_shapes0
.:,                           А2
dropout/mulЪ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*B
_output_shapes0
.:,                           А2
dropout/CastХ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*B
_output_shapes0
.:,                           А2
dropout/mul_1А
IdentityIdentitydropout/mul_1:z:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           А:& "
 
_user_specified_nameinputs
щ
f
J__inference_leaky_re_lu_91_layer_call_and_return_conditional_losses_555919

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
╗$
Э
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_554520

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_554505
assignmovingavg_1_554512
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
loc:@AssignMovingAvg/554505*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xп
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/554505*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_554505*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp╠
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/554505*
_output_shapes
:@2
AssignMovingAvg/sub_1╡
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/554505*
_output_shapes
:@2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_554505AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/554505*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpд
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/554512*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╖
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/554512*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_554512*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╪
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/554512*
_output_shapes
:@2
AssignMovingAvg_1/sub_1┐
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/554512*
_output_shapes
:@2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_554512AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/554512*
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
О
e
F__inference_dropout_37_layer_call_and_return_conditional_losses_554995

inputs
identityИa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/random_uniform/max╬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*A
_output_shapes/
-:+                           @*
dtype02&
$dropout/random_uniform/RandomUniformк
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub┌
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*A
_output_shapes/
-:+                           @2
dropout/random_uniform/mul╚
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv╗
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/GreaterEqualК
dropout/mulMulinputsdropout/truediv:z:0*
T0*A
_output_shapes/
-:+                           @2
dropout/mulЩ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                           @2
dropout/CastФ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                           @2
dropout/mul_1
IdentityIdentitydropout/mul_1:z:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:& "
 
_user_specified_nameinputs
й
G
+__inference_dropout_37_layer_call_fn_556078

inputs
identity╦
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
CPU2*0J 8*O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_5550002
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
╟
л
*__inference_conv2d_38_layer_call_fn_554578

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_38_layer_call_and_return_conditional_losses_5545702
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ё
А
7__inference_batch_normalization_95_layer_call_fn_556024

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
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_5545202
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
ь
f
J__inference_leaky_re_lu_93_layer_call_and_return_conditional_losses_555044

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
В
ї
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_554551

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
щ
f
J__inference_leaky_re_lu_92_layer_call_and_return_conditional_losses_556038

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
┤
d
+__inference_dropout_36_layer_call_fn_555954

inputs
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
CPU2*0J 8*O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_5549182
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Д
а
.__inference_sequential_24_layer_call_fn_555840

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
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identityИвStatefulPartitionedCall╖	
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_5553702
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ш
_input_shapesЖ
Г:         ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
є

▐
E__inference_conv2d_36_layer_call_and_return_conditional_losses_554266

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddп
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Н
ї
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_554703

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
ё
А
7__inference_batch_normalization_94_layer_call_fn_555905

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
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_5543682
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
 
_user_specified_nameinputs"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
C
input_18
serving_default_input_1:0         <
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:дч
╢g
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
	variables
trainable_variables
regularization_losses
	keras_api

signatures
╩__call__
╦_default_save_signature
+╠&call_and_return_all_conditional_losses"Еd
_tf_keras_sequentialцc{"class_name": "Sequential", "name": "sequential_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_24", "layers": [{"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_12", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "batch_input_shape": [null, 56, 56, 1], "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_94", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_91", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dropout", "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_95", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_92", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dropout", "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_96", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_93", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dropout", "config": {"name": "dropout_38", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_97", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_94", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dense", "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_24", "layers": [{"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_12", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "batch_input_shape": [null, 56, 56, 1], "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_94", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_91", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dropout", "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_95", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_92", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dropout", "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_96", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_93", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dropout", "config": {"name": "dropout_38", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_97", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_94", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Dense", "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
ё
	variables
regularization_losses
trainable_variables
	keras_api
═__call__
+╬&call_and_return_all_conditional_losses"р
_tf_keras_layer╞{"class_name": "UpSampling2D", "name": "up_sampling2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "up_sampling2d_12", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ж

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
╧__call__
+╨&call_and_return_all_conditional_losses" 
_tf_keras_layerх{"class_name": "Conv2D", "name": "conv2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 56, 56, 1], "config": {"name": "conv2d_36", "trainable": true, "batch_input_shape": [null, 56, 56, 1], "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
╖
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(regularization_losses
)trainable_variables
*	keras_api
╤__call__
+╥&call_and_return_all_conditional_losses"с
_tf_keras_layer╟{"class_name": "BatchNormalization", "name": "batch_normalization_94", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_94", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
м
+	variables
,regularization_losses
-trainable_variables
.	keras_api
╙__call__
+╘&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "LeakyReLU", "name": "leaky_re_lu_91", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_91", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
│
/	variables
0regularization_losses
1trainable_variables
2	keras_api
╒__call__
+╓&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"class_name": "Dropout", "name": "dropout_36", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
з

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
╫__call__
+╪&call_and_return_all_conditional_losses"А
_tf_keras_layerц{"class_name": "Conv2D", "name": "conv2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 28, 28, 1], "config": {"name": "conv2d_37", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
╖
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>	variables
?regularization_losses
@trainable_variables
A	keras_api
┘__call__
+┌&call_and_return_all_conditional_losses"с
_tf_keras_layer╟{"class_name": "BatchNormalization", "name": "batch_normalization_95", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_95", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
м
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
█__call__
+▄&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "LeakyReLU", "name": "leaky_re_lu_92", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_92", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
│
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
▌__call__
+▐&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"class_name": "Dropout", "name": "dropout_37", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
є

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
▀__call__
+р&call_and_return_all_conditional_losses"╠
_tf_keras_layer▓{"class_name": "Conv2D", "name": "conv2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
╕
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
с__call__
+т&call_and_return_all_conditional_losses"т
_tf_keras_layer╚{"class_name": "BatchNormalization", "name": "batch_normalization_96", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_96", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
м
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "LeakyReLU", "name": "leaky_re_lu_93", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_93", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
│
]	variables
^regularization_losses
_trainable_variables
`	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"class_name": "Dropout", "name": "dropout_38", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_38", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
┤
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
°

ekernel
fbias
g	variables
hregularization_losses
itrainable_variables
j	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"╤
_tf_keras_layer╖{"class_name": "Dense", "name": "dense_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6272}}}}
╢
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"р
_tf_keras_layer╞{"class_name": "BatchNormalization", "name": "batch_normalization_97", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_97", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 4}}}}
м
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "LeakyReLU", "name": "leaky_re_lu_94", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_94", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ў

xkernel
ybias
z	variables
{regularization_losses
|trainable_variables
}	keras_api
я__call__
+Ё&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "Dense", "name": "dense_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}}
ц
0
1
#2
$3
%4
&5
36
47
:8
;9
<10
=11
J12
K13
Q14
R15
S16
T17
e18
f19
l20
m21
n22
o23
x24
y25"
trackable_list_wrapper
ж
0
1
#2
$3
34
45
:6
;7
J8
K9
Q10
R11
e12
f13
l14
m15
x16
y17"
trackable_list_wrapper
 "
trackable_list_wrapper
╜
	variables
~metrics
trainable_variables

layers
regularization_losses
Аnon_trainable_variables
 Бlayer_regularization_losses
╩__call__
╦_default_save_signature
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
-
ёserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
	variables
Вmetrics
regularization_losses
Гlayers
trainable_variables
Дnon_trainable_variables
 Еlayer_regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
8:6@2sequential_24/conv2d_36/kernel
*:(@2sequential_24/conv2d_36/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
б
	variables
Жmetrics
regularization_losses
Зlayers
 trainable_variables
Иnon_trainable_variables
 Йlayer_regularization_losses
╧__call__
+╨&call_and_return_all_conditional_losses
'╨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
8:6@2*sequential_24/batch_normalization_94/gamma
7:5@2)sequential_24/batch_normalization_94/beta
@:>@ (20sequential_24/batch_normalization_94/moving_mean
D:B@ (24sequential_24/batch_normalization_94/moving_variance
<
#0
$1
%2
&3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
б
'	variables
Кmetrics
(regularization_losses
Лlayers
)trainable_variables
Мnon_trainable_variables
 Нlayer_regularization_losses
╤__call__
+╥&call_and_return_all_conditional_losses
'╥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
+	variables
Оmetrics
,regularization_losses
Пlayers
-trainable_variables
Рnon_trainable_variables
 Сlayer_regularization_losses
╙__call__
+╘&call_and_return_all_conditional_losses
'╘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
/	variables
Тmetrics
0regularization_losses
Уlayers
1trainable_variables
Фnon_trainable_variables
 Хlayer_regularization_losses
╒__call__
+╓&call_and_return_all_conditional_losses
'╓"call_and_return_conditional_losses"
_generic_user_object
8:6@@2sequential_24/conv2d_37/kernel
*:(@2sequential_24/conv2d_37/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
б
5	variables
Цmetrics
6regularization_losses
Чlayers
7trainable_variables
Шnon_trainable_variables
 Щlayer_regularization_losses
╫__call__
+╪&call_and_return_all_conditional_losses
'╪"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
8:6@2*sequential_24/batch_normalization_95/gamma
7:5@2)sequential_24/batch_normalization_95/beta
@:>@ (20sequential_24/batch_normalization_95/moving_mean
D:B@ (24sequential_24/batch_normalization_95/moving_variance
<
:0
;1
<2
=3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
б
>	variables
Ъmetrics
?regularization_losses
Ыlayers
@trainable_variables
Ьnon_trainable_variables
 Эlayer_regularization_losses
┘__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
B	variables
Юmetrics
Cregularization_losses
Яlayers
Dtrainable_variables
аnon_trainable_variables
 бlayer_regularization_losses
█__call__
+▄&call_and_return_all_conditional_losses
'▄"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
F	variables
вmetrics
Gregularization_losses
гlayers
Htrainable_variables
дnon_trainable_variables
 еlayer_regularization_losses
▌__call__
+▐&call_and_return_all_conditional_losses
'▐"call_and_return_conditional_losses"
_generic_user_object
9:7@А2sequential_24/conv2d_38/kernel
+:)А2sequential_24/conv2d_38/bias
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
б
L	variables
жmetrics
Mregularization_losses
зlayers
Ntrainable_variables
иnon_trainable_variables
 йlayer_regularization_losses
▀__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
9:7А2*sequential_24/batch_normalization_96/gamma
8:6А2)sequential_24/batch_normalization_96/beta
A:?А (20sequential_24/batch_normalization_96/moving_mean
E:CА (24sequential_24/batch_normalization_96/moving_variance
<
Q0
R1
S2
T3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
б
U	variables
кmetrics
Vregularization_losses
лlayers
Wtrainable_variables
мnon_trainable_variables
 нlayer_regularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
Y	variables
оmetrics
Zregularization_losses
пlayers
[trainable_variables
░non_trainable_variables
 ▒layer_regularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
]	variables
▓metrics
^regularization_losses
│layers
_trainable_variables
┤non_trainable_variables
 ╡layer_regularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
a	variables
╢metrics
bregularization_losses
╖layers
ctrainable_variables
╕non_trainable_variables
 ╣layer_regularization_losses
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
0:.	А12sequential_24/dense_43/kernel
):'2sequential_24/dense_43/bias
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
б
g	variables
║metrics
hregularization_losses
╗layers
itrainable_variables
╝non_trainable_variables
 ╜layer_regularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
8:62*sequential_24/batch_normalization_97/gamma
7:52)sequential_24/batch_normalization_97/beta
@:> (20sequential_24/batch_normalization_97/moving_mean
D:B (24sequential_24/batch_normalization_97/moving_variance
<
l0
m1
n2
o3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
б
p	variables
╛metrics
qregularization_losses
┐layers
rtrainable_variables
└non_trainable_variables
 ┴layer_regularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
t	variables
┬metrics
uregularization_losses
├layers
vtrainable_variables
─non_trainable_variables
 ┼layer_regularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
/:-2sequential_24/dense_44/kernel
):'2sequential_24/dense_44/bias
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
б
z	variables
╞metrics
{regularization_losses
╟layers
|trainable_variables
╚non_trainable_variables
 ╔layer_regularization_losses
я__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
X
%0
&1
<2
=3
S4
T5
n6
o7"
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
%0
&1"
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
<0
=1"
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
S0
T1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
n0
o1"
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
.__inference_sequential_24_layer_call_fn_555840
.__inference_sequential_24_layer_call_fn_555809
.__inference_sequential_24_layer_call_fn_555399
.__inference_sequential_24_layer_call_fn_555320└
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
ч2ф
!__inference__wrapped_model_554235╛
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
annotationsк *.в+
)К&
input_1         
Є2я
I__inference_sequential_24_layer_call_and_return_conditional_losses_555778
I__inference_sequential_24_layer_call_and_return_conditional_losses_555192
I__inference_sequential_24_layer_call_and_return_conditional_losses_555240
I__inference_sequential_24_layer_call_and_return_conditional_losses_555653└
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
Щ2Ц
1__inference_up_sampling2d_12_layer_call_fn_554254р
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
annotationsк *@в=
;К84                                    
┤2▒
L__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_554248р
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
annotationsк *@в=
;К84                                    
Й2Ж
*__inference_conv2d_36_layer_call_fn_554274╫
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
2К/+                           
д2б
E__inference_conv2d_36_layer_call_and_return_conditional_losses_554266╫
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
2К/+                           
м2й
7__inference_batch_normalization_94_layer_call_fn_555905
7__inference_batch_normalization_94_layer_call_fn_555914┤
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
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_555896
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_555874┤
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
/__inference_leaky_re_lu_91_layer_call_fn_555924в
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
J__inference_leaky_re_lu_91_layer_call_and_return_conditional_losses_555919в
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
+__inference_dropout_36_layer_call_fn_555959
+__inference_dropout_36_layer_call_fn_555954┤
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
╩2╟
F__inference_dropout_36_layer_call_and_return_conditional_losses_555949
F__inference_dropout_36_layer_call_and_return_conditional_losses_555944┤
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
Й2Ж
*__inference_conv2d_37_layer_call_fn_554426╫
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
д2б
E__inference_conv2d_37_layer_call_and_return_conditional_losses_554418╫
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
м2й
7__inference_batch_normalization_95_layer_call_fn_556024
7__inference_batch_normalization_95_layer_call_fn_556033┤
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
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_555993
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_556015┤
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
/__inference_leaky_re_lu_92_layer_call_fn_556043в
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
J__inference_leaky_re_lu_92_layer_call_and_return_conditional_losses_556038в
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
+__inference_dropout_37_layer_call_fn_556073
+__inference_dropout_37_layer_call_fn_556078┤
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
╩2╟
F__inference_dropout_37_layer_call_and_return_conditional_losses_556068
F__inference_dropout_37_layer_call_and_return_conditional_losses_556063┤
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
Й2Ж
*__inference_conv2d_38_layer_call_fn_554578╫
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
д2б
E__inference_conv2d_38_layer_call_and_return_conditional_losses_554570╫
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
м2й
7__inference_batch_normalization_96_layer_call_fn_556143
7__inference_batch_normalization_96_layer_call_fn_556152┤
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
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_556112
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_556134┤
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
/__inference_leaky_re_lu_93_layer_call_fn_556162в
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
J__inference_leaky_re_lu_93_layer_call_and_return_conditional_losses_556157в
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
+__inference_dropout_38_layer_call_fn_556197
+__inference_dropout_38_layer_call_fn_556192┤
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
╩2╟
F__inference_dropout_38_layer_call_and_return_conditional_losses_556182
F__inference_dropout_38_layer_call_and_return_conditional_losses_556187┤
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
╒2╥
+__inference_flatten_12_layer_call_fn_556214в
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
Ё2э
F__inference_flatten_12_layer_call_and_return_conditional_losses_556209в
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
)__inference_dense_43_layer_call_fn_556231в
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
D__inference_dense_43_layer_call_and_return_conditional_losses_556224в
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
7__inference_batch_normalization_97_layer_call_fn_556302
7__inference_batch_normalization_97_layer_call_fn_556311┤
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
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_556270
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_556293┤
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
/__inference_leaky_re_lu_94_layer_call_fn_556321в
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
J__inference_leaky_re_lu_94_layer_call_and_return_conditional_losses_556316в
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
)__inference_dense_44_layer_call_fn_556339в
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
D__inference_dense_44_layer_call_and_return_conditional_losses_556332в
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
3B1
$__inference_signature_wrapper_555431input_1▒
!__inference__wrapped_model_554235Л#$%&34:;<=JKQRSTefnomlxy8в5
.в+
)К&
input_1         
к "3к0
.
output_1"К
output_1         э
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_555874Ц#$%&MвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ э
R__inference_batch_normalization_94_layer_call_and_return_conditional_losses_555896Ц#$%&MвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ ┼
7__inference_batch_normalization_94_layer_call_fn_555905Й#$%&MвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @┼
7__inference_batch_normalization_94_layer_call_fn_555914Й#$%&MвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @э
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_555993Ц:;<=MвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ э
R__inference_batch_normalization_95_layer_call_and_return_conditional_losses_556015Ц:;<=MвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ ┼
7__inference_batch_normalization_95_layer_call_fn_556024Й:;<=MвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @┼
7__inference_batch_normalization_95_layer_call_fn_556033Й:;<=MвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @я
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_556112ШQRSTNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ я
R__inference_batch_normalization_96_layer_call_and_return_conditional_losses_556134ШQRSTNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ ╟
7__inference_batch_normalization_96_layer_call_fn_556143ЛQRSTNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А╟
7__inference_batch_normalization_96_layer_call_fn_556152ЛQRSTNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╕
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_556270bnoml3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ ╕
R__inference_batch_normalization_97_layer_call_and_return_conditional_losses_556293bnoml3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ Р
7__inference_batch_normalization_97_layer_call_fn_556302Unoml3в0
)в&
 К
inputs         
p
к "К         Р
7__inference_batch_normalization_97_layer_call_fn_556311Unoml3в0
)в&
 К
inputs         
p 
к "К         ┌
E__inference_conv2d_36_layer_call_and_return_conditional_losses_554266РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           @
Ъ ▓
*__inference_conv2d_36_layer_call_fn_554274ГIвF
?в<
:К7
inputs+                           
к "2К/+                           @┌
E__inference_conv2d_37_layer_call_and_return_conditional_losses_554418Р34IвF
?в<
:К7
inputs+                           @
к "?в<
5К2
0+                           @
Ъ ▓
*__inference_conv2d_37_layer_call_fn_554426Г34IвF
?в<
:К7
inputs+                           @
к "2К/+                           @█
E__inference_conv2d_38_layer_call_and_return_conditional_losses_554570СJKIвF
?в<
:К7
inputs+                           @
к "@в=
6К3
0,                           А
Ъ │
*__inference_conv2d_38_layer_call_fn_554578ДJKIвF
?в<
:К7
inputs+                           @
к "3К0,                           Ан
D__inference_dense_43_layer_call_and_return_conditional_losses_556224eef8в5
.в+
)К&
inputs                  
к "%в"
К
0         
Ъ Е
)__inference_dense_43_layer_call_fn_556231Xef8в5
.в+
)К&
inputs                  
к "К         д
D__inference_dense_44_layer_call_and_return_conditional_losses_556332\xy/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ |
)__inference_dense_44_layer_call_fn_556339Oxy/в,
%в"
 К
inputs         
к "К         █
F__inference_dropout_36_layer_call_and_return_conditional_losses_555944РMвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ █
F__inference_dropout_36_layer_call_and_return_conditional_losses_555949РMвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ │
+__inference_dropout_36_layer_call_fn_555954ГMвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @│
+__inference_dropout_36_layer_call_fn_555959ГMвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @█
F__inference_dropout_37_layer_call_and_return_conditional_losses_556063РMвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ █
F__inference_dropout_37_layer_call_and_return_conditional_losses_556068РMвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ │
+__inference_dropout_37_layer_call_fn_556073ГMвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @│
+__inference_dropout_37_layer_call_fn_556078ГMвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @▌
F__inference_dropout_38_layer_call_and_return_conditional_losses_556182ТNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ ▌
F__inference_dropout_38_layer_call_and_return_conditional_losses_556187ТNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ ╡
+__inference_dropout_38_layer_call_fn_556192ЕNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А╡
+__inference_dropout_38_layer_call_fn_556197ЕNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╞
F__inference_flatten_12_layer_call_and_return_conditional_losses_556209|JвG
@в=
;К8
inputs,                           А
к ".в+
$К!
0                  
Ъ Ю
+__inference_flatten_12_layer_call_fn_556214oJвG
@в=
;К8
inputs,                           А
к "!К                  █
J__inference_leaky_re_lu_91_layer_call_and_return_conditional_losses_555919МIвF
?в<
:К7
inputs+                           @
к "?в<
5К2
0+                           @
Ъ ▓
/__inference_leaky_re_lu_91_layer_call_fn_555924IвF
?в<
:К7
inputs+                           @
к "2К/+                           @█
J__inference_leaky_re_lu_92_layer_call_and_return_conditional_losses_556038МIвF
?в<
:К7
inputs+                           @
к "?в<
5К2
0+                           @
Ъ ▓
/__inference_leaky_re_lu_92_layer_call_fn_556043IвF
?в<
:К7
inputs+                           @
к "2К/+                           @▌
J__inference_leaky_re_lu_93_layer_call_and_return_conditional_losses_556157ОJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╡
/__inference_leaky_re_lu_93_layer_call_fn_556162БJвG
@в=
;К8
inputs,                           А
к "3К0,                           Аж
J__inference_leaky_re_lu_94_layer_call_and_return_conditional_losses_556316X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ ~
/__inference_leaky_re_lu_94_layer_call_fn_556321K/в,
%в"
 К
inputs         
к "К         ╙
I__inference_sequential_24_layer_call_and_return_conditional_losses_555192Е#$%&34:;<=JKQRSTefnomlxy@в=
6в3
)К&
input_1         
p

 
к "%в"
К
0         
Ъ ╙
I__inference_sequential_24_layer_call_and_return_conditional_losses_555240Е#$%&34:;<=JKQRSTefnomlxy@в=
6в3
)К&
input_1         
p 

 
к "%в"
К
0         
Ъ ╥
I__inference_sequential_24_layer_call_and_return_conditional_losses_555653Д#$%&34:;<=JKQRSTefnomlxy?в<
5в2
(К%
inputs         
p

 
к "%в"
К
0         
Ъ ╥
I__inference_sequential_24_layer_call_and_return_conditional_losses_555778Д#$%&34:;<=JKQRSTefnomlxy?в<
5в2
(К%
inputs         
p 

 
к "%в"
К
0         
Ъ к
.__inference_sequential_24_layer_call_fn_555320x#$%&34:;<=JKQRSTefnomlxy@в=
6в3
)К&
input_1         
p

 
к "К         к
.__inference_sequential_24_layer_call_fn_555399x#$%&34:;<=JKQRSTefnomlxy@в=
6в3
)К&
input_1         
p 

 
к "К         й
.__inference_sequential_24_layer_call_fn_555809w#$%&34:;<=JKQRSTefnomlxy?в<
5в2
(К%
inputs         
p

 
к "К         й
.__inference_sequential_24_layer_call_fn_555840w#$%&34:;<=JKQRSTefnomlxy?в<
5в2
(К%
inputs         
p 

 
к "К         ┐
$__inference_signature_wrapper_555431Ц#$%&34:;<=JKQRSTefnomlxyCв@
в 
9к6
4
input_1)К&
input_1         "3к0
.
output_1"К
output_1         я
L__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_554248ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_up_sampling2d_12_layer_call_fn_554254СRвO
HвE
CК@
inputs4                                    
к ";К84                                    