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
  Box,
  Divider,
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
import { VramCalculator } from "./components/VramCalculator";

const theme = createTheme({
  primaryColor: "cyan",
  colors: {
    cyan: [
      "#e6f7f7", "#c3ecec", "#9be1e1", "#7fd5d5",
      "#63c9c9", "#50bdbd", "#43a8a8", "#369393",
      "#2a7e7e", "#1e6969",
    ],
    dark: [
      "#F5F7FA", "#d0d4d8", "#a6adb5", "#7c8792",
      "#525f6e", "#3a4450", "#20323B", "#1a2a32",
      "#10151C", "#0a0e14",
    ],
  },
  fontFamily:
    "'Share Tech Mono', monospace",
  fontFamilyMonospace:
    "'Share Tech Mono', monospace",
  headings: {
    fontFamily: "'Orbitron', sans-serif",
    fontWeight: "700",
  },
  defaultRadius: "sm",
  components: {
    Paper: {
      styles: {
        root: { borderColor: "#20323B" },
      },
    },
    Card: {
      styles: {
        root: { borderColor: "#20323B" },
      },
    },
    Progress: {
      styles: {
        root: { backgroundColor: "#10151C" },
      },
    },
    Badge: {
      styles: {
        root: {
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        },
      },
    },
    Divider: {
      styles: {
        root: { borderColor: "#20323B" },
      },
    },
    Tabs: {
      styles: {
        tab: {
          fontFamily: "'Orbitron', sans-serif",
          fontWeight: 600,
        },
        list: { borderColor: "#20323B" },
      },
    },
    Switch: {
      styles: {
        track: { borderColor: "#7CB0C1" },
      },
    },
    Slider: {
      styles: {
        bar: { backgroundColor: "#7CB0C1" },
        thumb: { borderColor: "#50E3C2" },
      },
    },
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
            LLM Calc
          </Title>
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
            href="https://github.com/TeskesLab/llm-calculator"
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
          <Anchor
            href="https://github.com/TeskesLab/llm-calculator"
            underline="never"
            onClick={toggle}
          >
            GitHub
          </Anchor>
        </Stack>
      </Drawer>
    </Paper>
  );
}

export default function App() {
  return (
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <Notifications />
      <Box mih="100vh">
        <Header />

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

        <Container size="lg" pb={80} id="calculator">
          <VramCalculator />
        </Container>

        <Box component="footer" py="xl" ta="center">
          <Container size="md">
            <Divider mb="md" />
            <Text size="xs" c="dimmed">
              LLM Calc — VRAM & Performance Calculator.
              Based on apxml.com/tools/vram-calculator.
            </Text>
          </Container>
        </Box>
      </Box>
    </MantineProvider>
  );
}
