Python is a high-level, dynamically typed programming language known for its readability and simplicity. Here's an overview of Python's grammar and some practical methods and concepts:<br>

### Python Grammar:
Comments: Comments start with a hash symbol (#) and are ignored by the Python interpreter. They are used for adding explanations and notes within your code.<br>

### python
```python
# This is a comment
```
Variables and Data Types:<br>
Variables are used to store data. In Python, you don't need to declare the type of a variable explicitly; it's determined dynamically.<br>

Common data types include integers, floating-point numbers, strings, lists, tuples, dictionaries, sets, and more.<br>

### python
```python
x = 42  # Integer
y = 3.14  # Float
name = "John"  # String
my_list = [1, 2, 3]  # List
my_dict = {"key": "value"}  # Dictionary
```

Indentation:<br>
Python uses indentation to define code blocks (instead of curly braces or other delimiters). Indentation must be consistent within a block.<br>

### python
```python
if condition:
    # Indented block
    do_something()
else:
    # Another indented block
    do_something_else()
```

Conditionals:<br>
if, elif, and else are used for conditional statements.<br>

### python
```python
if condition:
    do_something()
elif another_condition:
    do_something_else()
else:
    do_another_thing()
```

Loops:<br>
for and while loops are used for iteration.<br>

### python
```python
for item in iterable:
    process_item()

while condition:
    do_something()
```
Functions:<br>
Functions are defined using the def keyword and can accept arguments and return values.<br>

### python
```python
def my_function(arg1, arg2):
    # Function body
    return result
```

Lists and Indexing:<br>
Lists are ordered collections of items. Indexing starts from 0.<br>
```python
my_list = [1, 2, 3, 4, 5]
first_element = my_list[0]
```

Strings:<br>
Strings are sequences of characters. You can perform various string operations, like slicing and concatenation.<br>
```python
my_string = "Hello, world!"
substring = my_string[0:5]  # Slicing
concatenated = "Hello" + " " + "world!"  # Concatenation
```
Dictionaries:<br>

Dictionaries are collections of key-value pairs.<br>
```python
my_dict = {"key1": "value1", "key2": "value2"}
value = my_dict["key1"]
```

Classes and Objects:<br>

Python supports object-oriented programming. You can define classes and create objects from them.<br>
```python
class MyClass:
    def __init__(self, param):
        self.data = param

obj = MyClass("Some data")
```
Useful Python Methods and Concepts:<br>
Built-in Functions: Python provides a wide range of built-in functions, such as len(), type(), print(), range(), and input().<br>

List Comprehensions: A concise way to create lists based on existing lists.<br>

```python
squares = [x**2 for x in range(1, 6)]
```
Slicing: Used to extract parts of sequences like strings and lists.<br>

```python
text = "Hello, world!"
sub_text = text[0:5]  # Extracts "Hello"
```
Object-Oriented Programming (OOP): Python supports classes, objects, and inheritance.<br>
Modules and Packages: Python's modular structure allows you to organize code into reusable modules and packages.<br>
Exception Handling: Using try, except, finally, and raise to handle and raise exceptions.<br>
File Handling: Reading and writing files with functions like open(), read(), write(), and close().<br>
Importing Libraries: Importing external libraries and modules using import.<br>
Lambda Functions: Anonymous functions created using the lambda keyword.<br>
Generators: Special functions used to create iterators, often using the yield keyword.<br>
Decorators: Functions that modify the behavior of other functions.<br>
This is just an overview of Python's grammar and some practical methods and concepts. Python's extensive standard library offers many more features and functionalities for various tasks. To use these features effectively, you should explore Python's official documentation and practice writing code.<br>

```python
Sub UnmergeAndFillDown_PreserveBlanks()
    Dim rng As Range, c As Range, m As Range
    Dim v As Variant
    On Error GoTo Done

    If TypeName(Selection) <> "Range" Then
        MsgBox "処理したい列（または範囲）を選択してから実行してください。"
        Exit Sub
    End If

    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual

    Set rng = Selection

    For Each c In rng.Cells
        If c.MergeCells Then
            Set m = c.MergeArea
            ' 同じ結合ブロックを重複処理しない
            If c.Address = m.Cells(1, 1).Address Then
                v = m.Cells(1, 1).Value
                m.UnMerge
                If Len(v) > 0 Then
                    ' 値がある結合ブロックのみ埋める
                    m.Value = v
                Else
                    ' 値が空の結合は空白のまま
                    ' 何もしない
                End If
                ' 任意：中央揃えをかけ直したい場合は次行のコメントを外す
                ' m.HorizontalAlignment = xlCenter
            End If
        End If
    Next c

Done:
    Application.Calculation = xlCalculationAutomatic
    Application.ScreenUpdating = True
End Sub

```

```python
=LET(
  fy,'BIDLデータ'!A2, plant,'BIDLデータ'!B2, product,'BIDLデータ'!C2, biz,'BIDLデータ'!D2,
  mr, XMATCH(1, ('投入データ'!A:A=fy)*('投入データ'!F:F=plant)*('投入データ'!G:G=product)*('投入データ'!I:I=biz), 0),
  IF(ISNA(mr),"NO MATCH",
    LET(
      s1,'BIDLデータ'!I2, t1, INDEX('投入データ'!J:J, mr),
      s2,'BIDLデータ'!K2, t2, INDEX('投入データ'!K:K, mr),
      s3,'BIDLデータ'!L2, t3, INDEX('投入データ'!L:L, mr),
      s4,'BIDLデータ'!N2, t4, INDEX('投入データ'!M:M, mr),
      s5,'BIDLデータ'!O2, t5, INDEX('投入データ'!N:N, mr),
      s6,'BIDLデータ'!Q2, t6, INDEX('投入データ'!O:O, mr),
      s7,'BIDLデータ'!R2, t7, INDEX('投入データ'!P:P, mr),
      exact, AND(s1=t1,s2=t2,s3=t3,s4=t4,s5=t5,s6=t6,s7=t7),
      rounded, AND(
        ROUND(s1,5)=ROUND(t1,5),
        ROUND(s2,5)=ROUND(t2,5),
        ROUND(s3,5)=ROUND(t3,5),
        ROUND(s4,5)=ROUND(t4,5),
        ROUND(s5,5)=ROUND(t5,5),
        ROUND(s6,5)=ROUND(t6,5),
        ROUND(s7,5)=ROUND(t7,5)
      ),
      IF(exact,"OK",IF(rounded,"OK≈(rounded)","NG"))
    )
  )
)
```

```python
=LET(
  fy,'BIDLデータ'!A2, plant,'BIDLデータ'!B2, product,'BIDLデータ'!C2, biz,'BIDLデータ'!D2,
  mr, XMATCH(1, ('投入データ'!A:A=fy)*('投入データ'!F:F=plant)*('投入データ'!G:G=product)*('投入データ'!I:I=biz), 0),
  IF(ISNA(mr),"",
    LET(
      names, {"売上数量","売上高","変動費","利益","固定費","荒利","コスト"},
      s, { 'BIDLデータ'!I2,'BIDLデータ'!K2,'BIDLデータ'!L2,'BIDLデータ'!N2,'BIDLデータ'!O2,'BIDLデータ'!Q2,'BIDLデータ'!R2 },
      t, { INDEX('投入データ'!J:J,mr), INDEX('投入データ'!K:K,mr), INDEX('投入データ'!L:L,mr), INDEX('投入データ'!M:M,mr), INDEX('投入データ'!N:N,mr), INDEX('投入データ'!O:O,mr), INDEX('投入データ'!P:P,mr) },
      exactFlags, s=t,
      roundFlags, ROUND(s,5)=ROUND(t,5),
      need, (NOT(exactFlags))*roundFlags,
      TEXTJOIN(", ", TRUE, IF(need, names, ""))
    )
  )
)
```
