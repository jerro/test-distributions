import std.stdio, std.mathspecial, std.algorithm, std.range, std.conv, 
    std.traits, std.numeric, std.random, std.typecons, std.typetuple,
    std.parallelism, std.getopt;

void accuracy()
{
    enum nchunks = 100;
    enum samplesPerChunk = 10L ^^ 9 / nchunks;
    
    @property zero(){ return new long[800]; }
    
    static auto accumulator(long[] a, long[] b)
    {
        auto r = b.dup;
        r[] += a[];
        return r;
    }

    auto mapper(int i)
    {
        auto r = zero;
        auto rng = Xorshift128(unpredictableSeed);
        foreach(_; 0 .. samplesPerChunk)
        {
            //auto x = normal(8.0, 1.0, rng);
            auto x = exponential(0.5, rng);
            //x = x > 0 && x < 16 ? x : 0;
            r[cast(int)(50 * x)] ++;
        }
        return r;
    }

    auto chunks = iota(nchunks).map!mapper();
    auto r = taskPool.reduce!accumulator(zero, chunks);
  
    double sum = r.reduce!"a+b";
    auto d = new double[r.length];
    foreach(i, _; r)
        d[i] = r[i] / sum;
    writeln(d);
    stderr.writeln(sum);
}

void speed()
{
    auto nsamples = 1_000_000_000L;

    //auto rng = Mt19937(1);
    auto rng = Xorshift128(unpredictableSeed);

    auto sum = 0.0;
    foreach(i; 0 .. nsamples)
    {
        //auto x = cauchy(8.0, 0.2, rng); 
        //auto x = normal(8.0, 1.0, rng);
        auto x = exponential(0.5, rng);
        //auto x = unif!double(rng);
        //writeln(x);
        //dist[cast(int)(50 * x)] ++;
        sum += x; 
    }
    writeln(sum);
} 

void plot()()
{
    alias double T;
    //mixin Cauchy!T;
    mixin Normal!T;
    alias ZigguratState!(func, integ, deriv, 0.5, 32) Z;

    auto x = iota(1000).map!(a => 0.01 * a).array;
    auto y = x.map!func().array;
    
    auto layerx = (size_t i) =>
        i == 0 ? 0 :
        i == Z.nlayers ? Z.tailX : 
        i == Z.nlayers + 1 ? T.max : Z.layers[Z.nlayers - i].x;

    auto layery = (size_t i) => i == Z.nlayers + 1 ? 0 : func(layerx(i));
    
    auto ymin = new T[x.length];
    auto ymax = ymin.dup;

    int layer = 0;
    foreach(i, _; x)
    {
        if(x[i] > layerx(layer + 1))
            layer++;
        
        ymin[i] = layery(layer + 1);
        ymax[i] = layery(layer);
        
        writefln("%s\t%s\t%s\t%s", x[i], ymin[i], ymax[i], y[i]);
    }

    import plot2kill.all;
    
    Figure()
        .addPlot(LineGraph(x, y))
        .addPlot(LineGraph(x, ymin))
        .addPlot(LineGraph(x, ymax))
        .showAsMain();
}

void main(string[] args)
{
    bool testSpeed; 
    getopt(args, "s", &testSpeed);
    
    if(testSpeed)
        speed();
    else
        accuracy();
}

