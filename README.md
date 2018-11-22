
# Workshop Scala

![Scala](https://hazelcast.org/wp-content/uploads/2016/04/scala-logo.jpg)
<br>
<br>
Este Workshop foi feito utilizando o [jupyter notebook](http://jupyter.org/) com o [spylon kernel](https://github.com/Valassis-Digital-Media/spylon-kernel), que reúne o poder do [Scala](https://www.scala-lang.org/), [Apache Spark](https://spark.apache.org/) e [Python](https://www.python.org/) em um mesmo ambiente interativo de desenvolvimento e visualização.
<br>
<br>
Para utilizá-lo, é necessário ter todas estas componentes instaladas, [este tutorial](https://intellipaat.com/tutorial/spark-tutorial/downloading-spark-and-getting-started/) explica muito claramente a instalação do Scala e Spark, o restante pode ser visto pelos sites das respectivas ferramentas.
<br>

---

# String Interpolation

Inserir pedaços de código no meio de uma string <br>
Para colocar uma variável, basta usar `$` <br>
Caso deseja-se executar uma expressão, colocar entre `{  }`


```scala
val n1 = 2
val n2 = 3

println(s"String interpolation: n1 = $n1, n2 = $n2")
```

    String interpolation: n1 = 2, n2 = 3





    n1: Int = 2
    n2: Int = 3





```scala
println(s"magic! ${2 + 4}")
```

    magic! 6


# POO

### Construtor


```scala
class Vector2d(var x: Int, var y: Int)
```




    defined class Vector2d





```scala
var v1 = new Vector2d(1, 2)
println(v1.x)
println(v1.y)
```

    1
    2





    v1: Vector2d = Vector2d@2f8b1a3b




### Case Class

Classe simples, só pra fornecer uma estrutura de dados rápida, já tem alguns métodos prontos, como `toString` <br>
(imutável)


```scala
case class Carro(marca: String, modelo: String)
```




    defined class Carro





```scala
val uno = Carro("Fiat", "Uno")
uno
```




    uno: Carro = Carro(Fiat,Uno)
    res31: Carro = Carro(Fiat,Uno)




### Default Parameters

Valores <em>default</em> para parâmetros


```scala
class Ponto(var x: Int = 0, var y: Int = 0)
```




    defined class Ponto





```scala
var p1 = new Ponto(3, 4)
var p2 = new Ponto()
var p3 = new Ponto(5)

println(s"p1 = (${p1.x}, ${p1.y})")
println(s"p2 = (${p2.x}, ${p2.y})")
println(s"p3 = (${p3.x}, ${p3.y})")
```

    p1 = (3, 4)
    p2 = (0, 0)
    p3 = (5, 0)





    p1: Ponto = Ponto@733e9292
    p2: Ponto = Ponto@2e0f0e8
    p3: Ponto = Ponto@6ca0614c




### Traits

Quase igual <em>Interface</em> no java, mas podem ter atributos e métodos concretos


```scala
trait Dog {
    val latido = "AU AU"
    def latir() = println(latido)
}
```




    defined trait Dog





```scala
class Labrador extends Dog
class Poodle extends Dog
```




    defined class Labrador
    defined class Poodle





```scala
val cachorro1 = new Labrador()
val cachorro2 = new Poodle()

println("labrador diz: ")
cachorro1.latir()
println("poodle diz: ")
cachorro2.latir()
```

    labrador diz: 
    AU AU
    poodle diz: 
    AU AU





    cachorro1: Labrador = Labrador@c0c74bc
    cachorro2: Poodle = Poodle@641c5511




### Operator Overloading

Sobrecarga de operador


```scala
class Vector(var x: Int = 0, var y: Int = 0) {
    override def toString(): String = s"($x, $y)"
}
```




    defined class Vector





```scala
var v1 = new Vector(2, 3)
var v2 = new Vector(5, 6)

println(s"v1: $v1")
println(s"v2: $v2")
```

    v1: (2, 3)
    v2: (5, 6)





    v1: Vector = (2, 3)
    v2: Vector = (5, 6)





```scala
class Vector(var x: Int = 0, var y: Int = 0) {
    override def toString(): String = s"($x, $y)"
    def add(v: Vector): Vector = new Vector(x + v.x, y + v.y)
}
```




    defined class Vector





```scala
var v1 = new Vector(2, 3)
var v2 = new Vector(5, 6)

println(v1.add(v2))
```

    (7, 9)





    v1: Vector = (2, 3)
    v2: Vector = (5, 6)





```scala
var v1 = new Vector(2, 3)
var v2 = new Vector(5, 6)

println(v1 add v2)
```

    (7, 9)





    v1: Vector = (2, 3)
    v2: Vector = (5, 6)





```scala
class Vector(var x: Int = 0, var y: Int = 0) {
    override def toString(): String = s"($x, $y)"
    def +(v: Vector): Vector = new Vector(x + v.x, y + v.y)
}
```




    defined class Vector





```scala
var v1 = new Vector(2, 3)
var v2 = new Vector(5, 6)

println(v1 + v2)
```

    (7, 9)





    v1: Vector = (2, 3)
    v2: Vector = (5, 6)





```scala
class Vector(var x: Int = 0, var y: Int = 0) {
    override def toString(): String = s"($x, $y)"
    def +(v: Vector): Vector = new Vector(x + v.x, y + v.y)
    def -(v: Vector): Vector = new Vector(x - v.x, y - v.y)
    def *(v: Vector): Vector = new Vector(x * v.x, y * v.y)
    def /(v: Vector): Vector = new Vector(x / v.x, y / v.y)
    def ?:%*~(v: Vector): Vector = new Vector(x + 2 * v.x, y - 2 * v.y)
    def !/<(): Vector = new Vector(-100, -100)
}
```




    defined class Vector





```scala
var v1 = new Vector(2, 3)
var v2 = new Vector(5, 6)

println(v1 + v2)
println(v1 - v2)
println(v1 * v2)
println(v1 / v2)
println(v1 ?:%*~ v2)
println(v1 !/<)
```

    (7, 9)
    (-3, -3)
    (10, 18)
    (0, 0)
    (12, -9)
    (-100, -100)





    warning: there was one feature warning; re-run with -feature for details
    v1: Vector = (2, 3)
    v2: Vector = (5, 6)




### Singleton

Classe que só pode ser instanciada 1 vez


```scala
object Matematica {
    def soma(n1: Int, n2: Int): Int = n1 + n2
    def sub(n1: Int, n2: Int): Int = n1 - n2
    def mult(n1: Int, n2: Int): Int = n1 * n2
    def div(n1: Int, n2: Int): Int = n1 / n2
}
```




    defined object Matematica





```scala
println(Matematica.soma(2, 3))
println(Matematica.sub(9, 2))
println(Matematica.mult(5, 9))
println(Matematica.div(121, 11))
```

    5
    7
    45
    11


# Funcional

Funções devem ser curtas, cumprir somente 1 objetivo, ter parâmetros e retornos, e não ter efeitos colaterais (não mudar o valor de variáveis globais). <br>
Diz-se que em programação puramente funcional não se pode nem ao menos declarar variáveis, apenas constantes

### Recursão Tradicional


```scala
def fatorial(x: Int): Int = {
    if (x <= 1) 1
    else x * fatorial(x - 1)
}
```




    fatorial: (x: Int)Int





```scala
println(fatorial(3))
println(fatorial(4))
println(fatorial(5))
println(fatorial(10))
```

    6
    24
    120
    3628800


### Tail Recursion

Descarta o <em>stack</em> das iterações anteriores, economiza muuuuita memória <br>
(demonstrar no quadro)


```scala
def fatorial_tail(x: Int, total: Int = 1): Int = {
    if (x <= 1) total
    else fatorial_tail(x - 1, total * x)
}
```




    fatorial_tail: (x: Int, total: Int)Int





```scala
println(fatorial_tail(3))
println(fatorial_tail(4))
println(fatorial_tail(5))
println(fatorial_tail(10))
```

    6
    24
    120
    3628800


### Higher-order Functions

Funções que recebem funções como parâmetro ou retornam funções


```scala
def somar_intervalo(start: Int, finish: Int, total: Int = 0): Int = {
    if (start > finish) total
    else somar_intervalo(start + 1, finish, total + start)
}
```




    somar_intervalo: (start: Int, finish: Int, total: Int)Int





```scala
println(somar_intervalo(1, 2))
println(somar_intervalo(1, 3))
println(somar_intervalo(1, 4))
println(somar_intervalo(1, 8))
```

    3
    6
    10
    36



```scala
def somar_quadrados_dos_intervalos(start: Int, finish: Int, total: Int = 0): Int = {
    if (start > finish) total
    else somar_quadrados_dos_intervalos(start + 1, finish, total + start*start)
}
```




    somar_quadrados_dos_intervalos: (start: Int, finish: Int, total: Int)Int





```scala
println(somar_quadrados_dos_intervalos(1, 2))
println(somar_quadrados_dos_intervalos(1, 3))
println(somar_quadrados_dos_intervalos(1, 4))
println(somar_quadrados_dos_intervalos(1, 8))
```

    5
    14
    30
    204



```scala
def somar(start: Int, finish: Int, f: Int => Int, total: Int = 0): Int = {
    if (start > finish) total
    else somar(start + 1, finish, f, total + f(start))
}
```




    somar: (start: Int, finish: Int, f: Int => Int, total: Int)Int





```scala
def quadrado(num: Int): Int = num * num
def somar_quadrados(start: Int, finish: Int): Int = somar(start, finish, quadrado)
```




    quadrado: (num: Int)Int
    somar_quadrados: (start: Int, finish: Int)Int





```scala
println(somar_quadrados(1, 2))
println(somar_quadrados(1, 3))
println(somar_quadrados(1, 4))
println(somar_quadrados(1, 8))
```

    5
    14
    30
    204


### Anonymous Function

Função que é definida somente no contexto, não tem nome


```scala
def somar_cubo(start:Int, finish:Int):Int = somar(start, finish, x => x * x * x)
```




    somar_cubo: (start: Int, finish: Int)Int





```scala
println(somar_cubo(1, 2))
println(somar_cubo(1, 3))
println(somar_cubo(1, 4))
println(somar_cubo(1, 8))
```

    9
    36
    100
    1296


somar_quarta, por exemplo


```scala
println(somar(1, 2, x => x * x * x * x))
println(somar(1, 3, x => x * x * x * x))
println(somar(1, 4, x => x * x * x * x))
println(somar(1, 8, x => x * x * x * x))
```

    17
    98
    354
    8772


### Exemplos Com Listas

<em>map</em> e <em>filter</em>


```scala
var arr1 = Array(-3, -1, 0, 2, 3)
var arr2 = Array(-5, 2, 7, 10, 15)
```




    arr1: Array[Int] = Array(-3, -1, 0, 2, 3)
    arr2: Array[Int] = Array(-5, 2, 7, 10, 15)




#### Map

Passa todos os valores de uma lista por uma função


```scala
arr1.map(Math.abs)
```




    res96: Array[Int] = Array(3, 1, 0, 2, 3)




Obs.: Como no paradigma funcional, funções não devem ter efeitos colaterais (elas recebem parâmetros e retornam um valor, mas não interferem em valores globais), a função <em>map</em> simplesmente retorna um novo `Array`, mas não muda seu conteúdo


```scala
arr1
```




    res97: Array[Int] = Array(-3, -1, 0, 2, 3)





```scala
arr2.map(x => x + 3)
```




    res98: Array[Int] = Array(-2, 5, 10, 13, 18)




#### Filter

Retorna um novo `Array` somente com os valores que passarem no filtro


```scala
arr1.filter(x => x > 0)
```




    res99: Array[Int] = Array(2, 3)





```scala
arr2.filter(x => x % 2 == 0)
```




    res100: Array[Int] = Array(2, 10)




### Funções aninhadas

Voltando à classe Vector, para fazer um exponencial de vetor


```scala
class Vector(var x: Int = 0, var y: Int = 0) {
    override def toString(): String = s"($x, $y)"
    def +(v: Vector): Vector = new Vector(x + v.x, y + v.y)
    def -(v: Vector): Vector = new Vector(x - v.x, y - v.y)
    def *(v: Vector): Vector = new Vector(x * v.x, y * v.y)
    def /(v: Vector): Vector = new Vector(x / v.x, y / v.y)
    def ?:%*~(v: Vector): Vector = new Vector(x + 2 * v.x, y - 2 * v.y)
    def !/<(): Vector = new Vector(-100, -100)
    
    def **(v: Vector): Vector = {
        def exp(base: Int, e: Int, total: Int = 1): Int = {
            if (e == 0) total
            else exp(base, e - 1, total * base)
        }
        new Vector(exp(x, v.x), exp(y, v.y))
    }
}
```




    defined class Vector





```scala
var v1 = new Vector()
var v2 = new Vector(2, 3)
var v3 = new Vector(3, 4)

println(v1 ** v2)
println(v2 ** v1)
println(v2 ** v3)
```

    (0, 0)
    (1, 1)
    (8, 81)





    v1: Vector = (0, 0)
    v2: Vector = (2, 3)
    v3: Vector = (3, 4)




---

# Apache Spark Examples


```scala
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
```




    import org.apache.spark.{SparkConf, SparkContext}
    import org.apache.spark.mllib.linalg._
    import org.apache.spark.mllib.stat.Statistics
    import org.apache.spark.rdd.RDD
    import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
    import org.apache.spark.mllib.linalg.Vectors
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.SingularValueDecomposition
    import org.apache.spark.mllib.linalg.Vector
    import org.apache.spark.mllib.linalg.Vectors
    import org.apache.spark.mllib.linalg.distributed.RowMatrix




## Estimando pi por simulação de Monte Carlo

### Paralelizado pelo Apache Spark


```scala
%%time
val NUM_SAMPLES = 100000

val count = sc.parallelize(1 to NUM_SAMPLES).filter { _ =>
  val x = math.random
  val y = math.random
  x*x + y*y < 1
}.count()
println(s"Pi is roughly ${4.0 * count / NUM_SAMPLES}")
```

    Pi is roughly 3.14416
    Time: 0.20341801643371582 seconds.
    





    NUM_SAMPLES: Int = 100000
    count: Long = 78604




## Exemplo de busca de correlação

Claramente há correlação entre os valores de `seriesX` e `seriesY`, porém como o último valor de `seriesY` (555) está bastante diferente, o valor final cai


```scala
val seriesX: RDD[Double] = sc.parallelize(Array(1, 2, 3, 3, 5))  // a series
// must have the same number of partitions and cardinality as seriesX
val seriesY: RDD[Double] = sc.parallelize(Array(11, 22, 33, 33, 555))

val correlation: Double = Statistics.corr(seriesX, seriesY)
println(s"Correlation is: $correlation")

val data: RDD[Vector] = sc.parallelize(
  Seq(
    Vectors.dense(1.0, 10.0, 100.0),
    Vectors.dense(2.0, 20.0, 200.0),
    Vectors.dense(5.0, 33.0, 366.0))
)

// calculate the correlation matrix
val correlMatrix: Matrix = Statistics.corr(data)
println(correlMatrix.toString)
```

    Correlation is: 0.8500286768773001
    1.0                 0.978883465889473   0.9903895695275671  
    0.978883465889473   1.0                 0.9977483233986101  
    0.9903895695275671  0.9977483233986101  1.0                 





    seriesX: org.apache.spark.rdd.RDD[Double] = ParallelCollectionRDD[27] at parallelize at <console>:37
    seriesY: org.apache.spark.rdd.RDD[Double] = ParallelCollectionRDD[28] at parallelize at <console>:39
    correlation: Double = 0.8500286768773001
    data: org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector] = ParallelCollectionRDD[33] at parallelize at <console>:44
    correlMatrix: org.apache.spark.mllib.linalg.Matrix =
    1.0                 0.978883465889473   0.9903895695275671
    0.978883465889473   1.0                 0.9977483233986101
    0.9903895695275671  0.9977483233986101  1.0




## K-Means Clustering

https://en.wikipedia.org/wiki/K-means_clustering


```scala
// Load and parse the data
val data = sc.textFile("data/mllib/kmeans_data.txt")
val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

// Cluster the data into two classes using KMeans
val numClusters = 2
val numIterations = 20
val clusters = KMeans.train(parsedData, numClusters, numIterations)

// Evaluate clustering by computing Within Set Sum of Squared Errors
val WSSSE = clusters.computeCost(parsedData)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Save and load model
clusters.save(sc, "target/org/apache/spark/KMeansExample/KMeansModel")
val sameModel = KMeansModel.load(sc, "target/org/apache/spark/KMeansExample/KMeansModel")
```

    Within Set Sum of Squared Errors = 0.11999999999994547





    data: org.apache.spark.rdd.RDD[String] = data/mllib/kmeans_data.txt MapPartitionsRDD[37] at textFile at <console>:42
    parsedData: org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector] = MapPartitionsRDD[38] at map at <console>:43
    numClusters: Int = 2
    numIterations: Int = 20
    clusters: org.apache.spark.mllib.clustering.KMeansModel = org.apache.spark.mllib.clustering.KMeansModel@3aac0c56
    WSSSE: Double = 0.11999999999994547
    sameModel: org.apache.spark.mllib.clustering.KMeansModel = org.apache.spark.mllib.clustering.KMeansModel@17701690




## SVD - Singular Value Decomposition

https://en.wikipedia.org/wiki/Singular_value_decomposition


```scala
val data = Array(
  Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
  Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
  Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0))

val rows = sc.parallelize(data)

val mat: RowMatrix = new RowMatrix(rows)

// Compute the top 5 singular values and corresponding singular vectors.
val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(5, computeU = true)
val U: RowMatrix = svd.U  // The U factor is a RowMatrix.
val s: Vector = svd.s     // The singular values are stored in a local dense vector.
val V: Matrix = svd.V     // The V factor is a local dense matrix.

val collect = U.rows.collect()
println("U factor is:")
collect.foreach { vector => println(vector) }
println(s"Singular values are: $s")
println(s"V factor is:\n$V")
```

    2018-11-22 04:37:20 WARN  LAPACK:61 - Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
    2018-11-22 04:37:20 WARN  LAPACK:61 - Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK
    U factor is:
    [-0.38829130511665644,-0.9198099362554474,-0.056387441301709175,9.313225746154785E-9,0.0]
    [-0.5301719995198351,0.2730185511901228,-0.8027319114319463,0.0,0.0]
    [-0.7537556058139434,0.2817987790459642,0.5936682026454339,1.4901161193847656E-8,1.4901161193847656E-8]
    Singular values are: [13.029275535600473,5.368578733451684,2.5330498218813755,6.323166049206486E-8,2.0226934557075942E-8]
    V factor is:
    -0.31278534337232633   0.3116713569157832    ... (5 total)
    -0.029801450130953977  -0.17133211263608739  ...
    -0.12207248163673157   0.15256470925290191   ...
    -0.7184789931874109    -0.6809628499946365   ...
    -0.6084105917199364    0.6217072292290715    ...





    data: Array[org.apache.spark.mllib.linalg.Vector] = Array((5,[1,3],[1.0,7.0]), [2.0,0.0,3.0,4.0,5.0], [4.0,0.0,0.0,6.0,7.0])
    rows: org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector] = ParallelCollectionRDD[79] at parallelize at <console>:59
    mat: org.apache.spark.mllib.linalg.distributed.RowMatrix = org.apache.spark.mllib.linalg.distributed.RowMatrix@660974d9
    svd: org.apache.spark.mllib.linalg.SingularValueDecomposition[org.apache.spark.mllib.linalg.distributed.RowMatrix,org.apache.spark.mllib.linalg.Matrix] =
    SingularValueDecomposition(org.apache.spark.mllib.linalg.distributed.RowMatrix@5ecdf23,[13.029275535600473,5.368578733451684,2.5330498218813755,6.323166049206486E-8,2.0226934557075942E-8],-0.31278534337232633   0.3116713569157832    ... (5 total)
    -0.029801450130953977  ...


