module counts;
import std.range, std.algorithm;

struct Counts(int M)
{
    alias ushort TBuffered;
    enum log2BlockSize = 8 * TBuffered.sizeof;
    enum size_t blockSize = cast(size_t) 1 << log2BlockSize; 
    enum size_t bufferSize = blockSize * M;

    struct PointerPair
    {
        this(TBuffered[] arr)
        {
            start = arr.ptr;
            end = arr.ptr + arr.length; 
        }

        TBuffered* start;
        TBuffered* end;
    }

    uint[] counts;
    PointerPair[] buffers;

    this(size_t n)
    {
        counts = new uint[](n);
        buffers = bufferSize
            .repeat((n + blockSize - 1) / blockSize)
            .map!(a => PointerPair(new TBuffered[a]))
            .array;
    }

    void save(size_t i)
    {
        assert(i <= buffers.length);
        assert(buffers[i].start <= buffers[i].end);
        assert(buffers[i].start >= buffers[i].end - bufferSize);
        
        auto pp = buffers[i];
        auto start = pp.end - bufferSize;
        foreach(e; start[0 .. pp.start - start])
            counts[(i << log2BlockSize) + e] ++;

        buffers[i].start = start; 
    }

    void increment(size_t i)
    {
        assert(i <= counts.length);

        auto iblock = i >> log2BlockSize;
        auto start = buffers[iblock].start;
        auto end = buffers[iblock].end;
        *start = cast(TBuffered) i;
        start++;
        buffers[iblock].start = start;
        if(start == end)
            save(iblock);
    }

    auto opSlice()
    {
        foreach(i; 0 .. buffers.length)
            save(i);

        return counts;
    }
}

unittest
{
    import std.random;

    auto n = 1_299_827;
    
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
{
    import std.stdio;

    void increment(uint[] arr, size_t i)
    {
        arr[i]++;
    }

    void main()
    {
        enum n = 1 << 24;

        version(all)
            auto c = Counts!2(n);
        else
            auto c = new uint[](n);

        auto rand = 20935742;

        foreach(i; 0 .. 1000_000_000)
        {
            c.increment(rand & (n - 1));
            rand = rand * 1664525 + 1013904223; 
        }
        writeln(c[][c[][0] % n]);
    }
}
