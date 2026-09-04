package com.agentforge;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * AgentForge Interview - Application Tests
 */
class AppTest {
    
    @Test 
    void contextLoads() {
        // 验证应用主类存在
        assertNotNull(App.class);
    }
}
