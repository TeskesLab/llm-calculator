import { useEffect, useState } from "react";
import {
  MantineProvider,
  createTheme,
  Container,
  Title,
  Text,
  Paper,
  Group,
  Stack,
  Anchor,
  Card,
  Grid,
  Box,
  Divider,
  Badge,
  Burger,
  Drawer,
  ActionIcon,
  useMantineColorScheme,
  useComputedColorScheme,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { Notifications } from "@mantine/notifications";
import {
  IconSun,
  IconMoon,
  IconCalculator,
  IconBrandGithub,
} from "@tabler/icons-react";
import { initWasm, isWasmReady } from "./wasm-loader";
import { VramCalculator } from "./components/VramCalculator";

const theme = createTheme({
  primaryColor: "violet",
  fontFamily:
    "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
  defaultRadius: "md",
  headings: {
    fontWeight: "700",
  },
});

function Header() {
  const [opened, { toggle }] = useDisclosure(false);
  const { setColorScheme } = useMantineColorScheme();
  const computedColorScheme = useComputedColorScheme("light");

  return (
    <Paper
      component="header"
      pos="fixed"
      top={0}
      left={0}
      right={0}
      style={{ zIndex: 100 }}
      shadow="sm"
      px="md"
    >
      <Group h={60} justify="space-between" wrap="nowrap">
        <Group gap="xs">
          <IconCalculator size={28} stroke={1.5} />
          <Title order={4} style={{ whiteSpace: "nowrap" }}>
            AI Calc
          </Title>
        </Group>

        <Group gap="xs" visibleFrom="sm">
          <Anchor href="#calculator" underline="never" fz="sm" fw={500}>
            Calculator
          </Anchor>
          <Anchor href="#about" underline="never" fz="sm" fw={500}>
            About
          </Anchor>
          <Anchor href="#faq" underline="never" fz="sm" fw={500}>
            FAQ
          </Anchor>
        </Group>

        <Group gap="xs">
          <ActionIcon
            variant="default"
            size="lg"
            onClick={() =>
              setColorScheme(computedColorScheme === "dark" ? "light" : "dark")
            }
          >
            {computedColorScheme === "dark" ? (
              <IconSun size={18} />
            ) : (
              <IconMoon size={18} />
            )}
          </ActionIcon>
          <ActionIcon
            variant="default"
            size="lg"
            component="a"
            href="https://github.com"
            target="_blank"
          >
            <IconBrandGithub size={18} />
          </ActionIcon>
          <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
        </Group>
      </Group>

      <Drawer opened={opened} onClose={toggle} title="Menu" size="xs">
        <Stack gap="md">
          <Anchor href="#calculator" underline="never" onClick={toggle}>
            Calculator
          </Anchor>
          <Anchor href="#about" underline="never" onClick={toggle}>
            About
          </Anchor>
          <Anchor href="#faq" underline="never" onClick={toggle}>
            FAQ
          </Anchor>
        </Stack>
      </Drawer>
    </Paper>
  );
}

export default function App() {
  const [wasmReady, setWasmReady] = useState(false);
  const [wasmError, setWasmError] = useState<string | null>(null);

  useEffect(() => {
    if (isWasmReady()) {
      setWasmReady(true);
      return;
    }
    initWasm()
      .then(() => setWasmReady(true))
      .catch((err) => setWasmError(err.message || "Failed to load calculator engine"));
  }, []);

  return (
    <MantineProvider theme={theme} defaultColorScheme="auto">
      <Notifications />
      <Box bg="var(--mantine-color-body)" mih="100vh">
        <Header />

        {/* Hero Section */}
        <Box pt={100} pb={60} ta="center">
          <Container size="md">
            <Title order={1} fz={{ base: 28, sm: 40 }} fw={900} mb="md">
              LLM Inference: VRAM & Performance Calculator
            </Title>
            <Text c="dimmed" size="lg" maw={600} mx="auto">
              Calculate GPU VRAM requirements and estimate performance for
              running large language models locally or in the cloud.
            </Text>
          </Container>
        </Box>

        {/* Calculator Section */}
        <Container size="lg" pb={80} id="calculator">
          {wasmError ? (
            <Paper p="xl" withBorder ta="center">
              <Text c="red" mb="md">
                Failed to load calculation engine: {wasmError}
              </Text>
              <Text size="sm" c="dimmed">
                Please check your connection and reload the page.
              </Text>
            </Paper>
          ) : !wasmReady ? (
            <Paper p="xl" withBorder ta="center">
              <Text>Loading calculator engine...</Text>
            </Paper>
          ) : (
            <VramCalculator />
          )}
        </Container>

        {/* How Calculations Are Made */}
        <Container size="lg" pb={80} id="about">
          <Grid>
            <Grid.Col span={{ md: 6 }}>
              <Card withBorder h="100%">
                <Title order={4} mb="sm">
                  How Calculations Are Made
                </Title>
                <Text size="sm" c="dimmed">
                  Memory usage is estimated using models that factor in
                  architecture (parameters, layers, hidden dimensions, active
                  experts, etc.), quantization, sequence length, and batch size.
                  Performance estimates consider model/hardware analysis and
                  benchmarks, though benchmark accuracy varies. Results are
                  approximate.
                </Text>
              </Card>
            </Grid.Col>
            <Grid.Col span={{ md: 6 }}>
              <Card withBorder h="100%">
                <Title order={4} mb="sm">
                  Frequently Asked Questions
                </Title>
                <Stack gap="sm">
                  <Box>
                    <Text fw={600} size="sm">
                      How accurate is this calculator?
                    </Text>
                    <Text size="sm" c="dimmed">
                      This calculator provides a theoretical estimation for
                      capacity planning. Results may vary based on framework
                      optimizations, driver overhead, and hardware specifics.
                    </Text>
                  </Box>
                  <Box>
                    <Text fw={600} size="sm">
                      Why do MoE models use so much VRAM?
                    </Text>
                    <Text size="sm" c="dimmed">
                      All experts must reside in VRAM to enable fast switching,
                      even though only a subset are active per token. The main
                      benefit of MoE is reduced compute, not memory.
                    </Text>
                  </Box>
                  <Box>
                    <Text fw={600} size="sm">
                      Why is VRAM higher than Ollama?
                    </Text>
                    <Text size="sm" c="dimmed">
                      Ollama defaults to 4-bit quantization (Q4_K_M). This
                      calculator defaults to FP16. Select a lower quantization
                      for comparable estimates.
                    </Text>
                  </Box>
                </Stack>
              </Card>
            </Grid.Col>
          </Grid>
        </Container>

        {/* Recent Updates */}
        <Container size="lg" pb={80}>
          <Card withBorder>
            <Title order={4} mb="md">
              Recent Updates
            </Title>
            <Stack gap="xs">
              <Group gap="xs" wrap="nowrap">
                <Badge size="sm" variant="light">Apr 2026</Badge>
                <Text size="sm">Fix memory calculation with fine-tuning with gradient accumulation.</Text>
              </Group>
              <Group gap="xs" wrap="nowrap">
                <Badge size="sm" variant="light">Feb 2026</Badge>
                <Text size="sm">Improve batch size scaling for fine-tuning.</Text>
              </Group>
              <Group gap="xs" wrap="nowrap">
                <Badge size="sm" variant="light">Feb 2026</Badge>
                <Text size="sm">Add training cost estimation calculation.</Text>
              </Group>
              <Group gap="xs" wrap="nowrap">
                <Badge size="sm" variant="light">Dec 2025</Badge>
                <Text size="sm">Fix per-user speed calculation for queuing with concurrent users.</Text>
              </Group>
              <Group gap="xs" wrap="nowrap">
                <Badge size="sm" variant="light">Dec 2025</Badge>
                <Text size="sm">Fix TFTT and TPS calculations for MoE models and Flash Attention.</Text>
              </Group>
            </Stack>
          </Card>
        </Container>

        {/* Footer */}
        <Box component="footer" py="xl" ta="center">
          <Container size="md">
            <Divider mb="md" />
            <Group justify="center" gap="lg" mb="xs">
              <Anchor href="#about" size="sm" c="dimmed" underline="never">
                About
              </Anchor>
              <Anchor href="#faq" size="sm" c="dimmed" underline="never">
                FAQ
              </Anchor>
              <Anchor
                href="https://github.com"
                target="_blank"
                size="sm"
                c="dimmed"
                underline="never"
              >
                GitHub
              </Anchor>
            </Group>
            <Text size="xs" c="dimmed">
              AI Calc — LLM VRAM Calculator.
              Based on apxml.com/tools/vram-calculator.
            </Text>
          </Container>
        </Box>
      </Box>
    </MantineProvider>
  );
}
