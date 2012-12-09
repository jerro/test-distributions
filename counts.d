module counts;

import core.bitop, core.memory;

struct SubByteArray(uint bitsPerElem) 
if((bitsPerElem & (bitsPerElem - 1)) == 0)
{
    enum uint elemMask = (1 << bitsPerElem) - 1;
    enum uint elementsPerUInt = uint.sizeof * 8 / bitsPerElem;

    enum uint iShift = bsr(elementsPerUInt);
    enum uint iMask = elementsPerUInt - 1;

    uint* data;
    size_t n;
   
    this(size_t n)
    {
        this.n = n;
        auto len = (n >> iShift) + 1;
        data = cast(uint*) GC.malloc(len * uint.sizeof);
        data[0 .. len] = 0;
    }
 
    uint opIndex(size_t i)
    {
        assert(i < n);

        uint shift = (i & iMask) * bitsPerElem;
        return (data[i >> iShift] >> shift) & elemMask;
    }

    void opIndexAssign(uint val, size_t i)
    {
        assert(i < n);
        assert((val & ~elemMask) == 0);

        uint shift = (i & iMask) * bitsPerElem;
        auto p = &data[i >> iShift];
        *p = (*p & ~(elemMask << shift)) | (val << shift);
    }
}

import std.stdio, std.random, std.range, std.algorithm;

unittest
{
    static void test(int nbits)()
    {
        int n = 1009;

        auto a = new int[n];
        auto s = SubByteArray!nbits(n);

        foreach(_; 0 .. 10 * n)
        {
            auto i = uniform(0, n);
            a[i] = (a[i] + 1) & s.elemMask; 
            s[i] = (s[i] + 1) & s.elemMask; 
        }

        foreach(i; 0 .. n)
            assert(a[i] == s[i]);
    }

    test!1();
    test!2();
    test!4();
    test!8(); 
}

struct Counts(int bitsPerCachedElem)
{
    enum elementsPerCacheLine = 16;
    
    SubByteArray!bitsPerCachedElem cached;
    uint* data;
    size_t n;    

    this(size_t n)
    {
        this.n = n;

        auto len = (n / elementsPerCacheLine + 1) * elementsPerCacheLine;
        data = cast(uint*) GC.malloc(len * uint.sizeof);
        data[0 .. len] = 0; 
        cached = typeof(cached)(len);        

    }

    private void saveElement(size_t i)
    {
        data[i] += cached[i];
        cached[i] = 0;
    }

    void increment(size_t i)
    {
        assert(i < n);

        auto prev = cached[i];

        if(prev == cached.elemMask)
        {
            // save this cache line

            auto clStart = i & ~(elementsPerCacheLine - 1);
            foreach(j; 0 .. elementsPerCacheLine)
                saveElement(clStart + j); 
            
            prev = 0;
        }

        cached[i] = prev + 1;
    }

    auto opSlice()
    {
        foreach(i; 0 .. n)
            saveElement(i);

        return data[0 .. n];   
    }
}

unittest
{
    auto n = 1009;
    
    auto c = Counts!2(n);
    auto a = new int[n];

    foreach(_; 0 .. 10 * n)
    {
        auto i = uniform(0, n);
        a[i]++;
        c.increment(i);
    }

    foreach(aa, ee; zip(a, c[]))
        assert(aa == ee);
}

version(BenchCounts)
    void main()
    {
        enum n = 1 << 24;

        auto c = Counts!4(n);

        auto rand = 20935742;
        
        foreach(i; 0 .. 1000_000_000)
        {
            c.increment(rand & (n - 1));
            rand = rand * 1664525 + 1013904223; 
        }

        writeln(c[][c[][0] % n]);
    }
