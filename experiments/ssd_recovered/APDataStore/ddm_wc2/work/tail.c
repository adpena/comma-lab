uint64_t f26_total_frequency(void) {
    return F26_TOTAL;
}

void f26_hpac_last_timing(double output[5]) {
    if (output) memcpy(output, f26_last_timing, sizeof(f26_last_timing));
}
